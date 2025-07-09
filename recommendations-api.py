from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
import asyncpg
import asyncio
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractionData(BaseModel):
    user_id: str
    song_id: str
    timestamp: Optional[datetime] = None

class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = 10

class RecommendationResponse(BaseModel):
    song_id: str
    score: float

class TrainModelRequest(BaseModel):
    interactions: List[InteractionData]

class DatabaseConfig:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.pool = None

    async def create_pool(self):
        self.pool = await asyncpg.create_pool(self.database_url)

    async def close_pool(self):
        if self.pool:
            await self.pool.close()

db_config = DatabaseConfig()

class RecomendationEngine:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.user_id_map = {}
        self.song_id_map = {}
        self.reverse_user_map = {}
        self.reverse_song_map = {}
        self.interacations_df = None
        self.is_trained = False

    async def get_interactions_from_db(self):
        try:
            async with db_config.pool.acquire() as conn:
                listenings = await conn.fetch("""
                    SELECT "userId", "songId", timestamp
                    FROM "Listening"
                    ORDER BY timestamp DESC
                """)

                favorites = await conn.fetch("""
                    SELECT u.id as "userId", s.id as "songId", s."createdAt" as timestamp
                    FROM "User" u
                    JOIN "_favorite" f ON u.id = f."A"
                    JOIN "Song" s ON s.id = f."B"
                """)
                
                interactions = []

                for record in listenings:
                    interactions.append({
                        'user_id': record['userId'],
                        'song_id': record['songId'],
                        'weight': 1.0,
                        'timestamp': record['timestamp']
                    })

                for record in favorites:
                    interactions.append({
                        'user_id': record['userId'],
                        'song_id': record['songId'],
                        'weight': 2.0,
                        'timestamp': record['timestamp']
                    })
                
                return interactions
        except Exception as e:
            logger.error(f"Error fetching interactions from database: {e}")
            return []
        
    async def train_model(self, interactions_data: List[dict] = None):
        try:
            if interactions_data is None:
                interactions_data = await self.get_interactions_from_db()

            if not interactions_data:
                logger.warning("No interactions data found in database.")
                return False
            
            df = pd.DataFrame(interactions_data)

            if df.empty:
                logger.warning("Interactions DataFrame is empty.")
                return False
            
            self.interacations_df = df

            unique_users = df['user_id'].unique()
            unique_songs = df['song_id'].unique()

            self.user_id_map = {
                user_id: idx for idx, user_id in enumerate(unique_users)
            }
            self.song_id_map = {
                song_id: idx for idx, song_id in enumerate(unique_songs)
            }
            self.reverse_user_map = {
                idx: user_id for user_id, idx in self.user_id_map.items()
            }
            self.reverse_song_map = {
                idx: song_id for song_id, idx in self.song_id_map.items()
            }
            
            self.dataset = Dataset()
            self.dataset.fit(unique_users, unique_songs)
            
            interactions_list = []
            weight_list = []
            
            for _, row in df.iterrows():
                interactions_list.append((row['user_id'], row['song_id']))
                weight_list.append(row.get('weight', 1.0))
                
            interactions_matrix = self.dataset.build_interactions(
                [(x[0], x[1], x[2]) for x in zip(df['user_id'], df['song_id'], df['weight'])]
            )[0]
            
            self.model = LightFM(
                loss='logistic',
                learning_rate=0.01,
                item_alpha=1e-6,
                user_alpha=1e-6
            )
            
            self.model.fit(
                interactions_matrix,
                epochs=5,
                num_threads=1,
                verbose=True
            )
            
            self.is_trained = True
            logger.info("Model trained successfully.")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
        
    def get_recommendations(self, user_id: str, limit: int = 10) -> List[RecommendationResponse]:
        logger.info("=== GET_RECOMMENDATIONS CALLED ===")
        logger.info(f"Model trained? {self.is_trained}")
        logger.info(f"Request user_id: {user_id}")
        logger.info(f"Known users: {list(self.user_id_map.keys())}")

        if not self.is_trained:
            logger.warning("Model is not trained yet.")
            return []

        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not found in user_id_map.")
            return []

        user_idx = self.user_id_map[user_id]
        n_songs = len(self.song_id_map)
        scores = self.model.predict(user_idx, np.arange(n_songs))

        user_songs = set(self.interacations_df[
            self.interacations_df['user_id'] == user_id
        ]['song_id'].tolist())
        logger.info(f"User {user_id} interactions: {user_songs}")

        recommendations = []
        for song_idx, score in enumerate(scores):
            song_id = self.reverse_song_map[song_idx]
            if song_id not in user_songs:
                recommendations.append(RecommendationResponse(song_id=song_id, score=float(score)))

        recommendations.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Returning {len(recommendations[:limit])} recommendations")
        return recommendations[:limit]
            
    def get_simmilar_songs(self, song_id: str, limit: int = 10) -> List[RecommendationResponse]:
        try:
            if not self.is_trained or song_id not in self.song_id_map:
                return []
            
            song_idx = self.song_id_map[song_id]
            
            song_embeddings = self.model.item_embeddings
            target_embedding = song_embeddings[song_idx]
            
            similarities = np.dot(song_embeddings, target_embedding) / (
                np.linalg.norm(song_embeddings, axis=1) * np.linalg.norm(target_embedding)
            )
            
            similar_indices = np.argsort(similarities)[::-1][1:limit+1]
            
            recommendations = []
            for idx in similar_indices:
                similar_song_id = self.reverse_song_map[idx]
                recommendations.append(RecommendationResponse(
                    song_id=similar_song_id,
                    score=similarities[idx]
                ))
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar songs: {e}")
            return []
        
recommendation_engine = RecomendationEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_config.create_pool()
    logger.info('База данных инициализирована')
    await recommendation_engine.train_model()
    logger.info("Модель обучена")
    
    yield
    
    await db_config.close_pool()
    logger.info('База данных закрыта')
    
app = FastAPI(title="Klix Reccomendations API", version="1.0.0", lifespan=lifespan)
    
@app.get('/')
async def root():
    return {"message": "KLIX RECOMMENDATIONS API", "version": "1.0.0"}

@app.get('/health')
async def health_check():
    return {
        "status": "Healthy",
        "model_trained": recommendation_engine.is_trained,
        "timestamp": datetime.now().isoformat()
    }

@app.post('/recommendations')
async def get_recommendations(request: RecommendationRequest) -> List[RecommendationResponse]:
    try:
        recommendations = recommendation_engine.get_recommendations(
            request.user_id,
            request.limit
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post('/similar-song/{song_id}')
async def get_similar_song(song_id: str, limit: int = 10) -> List[RecommendationResponse]:
    try:
        recommendations = recommendation_engine.get_simmilar_songs(song_id, limit)
        
        return recommendations
    except Exception as e:
        logger.error(f"Error getting similar song: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post('/train')
async def train_model(request: Optional[TrainModelRequest] = None):
    try:
        interactions_data = None
        if request and request.interactions:
            interactions_data = [
                {
                    "user_id": interaction.user_id,
                    "song_id": interaction.song_id,
                    "weight": 1.0,
                    "timestamp": interaction.timestamp or datetime.now()
                }
                for interaction in request.interactions
            ]
            
        success = await recommendation_engine.train_model(interactions_data)
        
        if success:
            return {"message": "Model trained successfully", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Failed to train model")
            
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/interactions")
async def record_interaction(interaction: InteractionData):
    try:
        async with db_config.pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO "Listening" ("userId", "songId", timestamp)
                VALUES ($1, $2, $3)
                ON CONFLICT ("userId", "songId") 
                DO UPDATE SET timestamp = $3
            """, interaction.user_id, interaction.song_id, 
                interaction.timestamp or datetime.now())
        
        return {"message": "Interaction recorded successfully"}
        
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)