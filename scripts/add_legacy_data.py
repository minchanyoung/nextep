#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실제 레거시 데이터를 ChromaDB에 추가하는 스크립트
"""

import chromadb
import os
import requests
import sys
sys.path.insert(0, os.path.abspath('.'))

from app.rag_data import LABOR_MARKET_TRENDS, LEARNING_RECOMMENDATIONS

def get_embedding(text, model="llama3:latest"):
    """Ollama API로 임베딩 생성"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"임베딩 API 오류: {response.status_code}")
            return None
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

def add_legacy_data():
    """레거시 데이터를 ChromaDB에 추가"""
    
    try:
        # ChromaDB 연결
        persist_dir = os.path.join('instance', 'chroma_db')
        client = chromadb.PersistentClient(path=persist_dir)
        
        # 기존 컬렉션 가져오기 또는 생성
        try:
            collection = client.get_collection("labor_market_docs")
            print("기존 컬렉션 사용")
        except:
            collection = client.create_collection(
                name="labor_market_docs",
                metadata={"hnsw:space": "cosine"}
            )
            print("새 컬렉션 생성")
        
        print(f"현재 문서 수: {collection.count()}")
        
        # 노동시장 트렌드 데이터 추가
        print(f"\n노동시장 트렌드 데이터 처리 중... ({len(LABOR_MARKET_TRENDS)}개)")
        
        embeddings = []
        texts = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(LABOR_MARKET_TRENDS):
            content = item.get("content", "")
            if not content.strip():
                continue
                
            print(f"  {i+1}. {item.get('title', 'No title')}")
            
            # 임베딩 생성
            embedding = get_embedding(content)
            if embedding:
                embeddings.append(embedding)
                texts.append(content)
                
                metadata = {
                    'source': 'legacy_labor_market',
                    'chunk_type': 'labor_trend',
                    'keywords': ','.join(item.get('keywords', [])),
                    'id': item.get('id', f"labor_trend_{i}"),
                    'title': item.get('title', ''),
                    'content_length': len(content)
                }
                metadatas.append(metadata)
                ids.append(f"labor_{item.get('id', i)}")
                
                print(f"     임베딩 차원: {len(embedding)}")
            else:
                print(f"     임베딩 실패")
        
        # 학습 추천 데이터 추가
        print(f"\n학습 추천 데이터 처리 중... ({len(LEARNING_RECOMMENDATIONS)}개)")
        
        for i, item in enumerate(LEARNING_RECOMMENDATIONS):
            content = item.get("description", "")
            if not content.strip():
                continue
                
            print(f"  {i+1}. {item.get('title', 'No title')}")
            
            # 임베딩 생성
            embedding = get_embedding(content)
            if embedding:
                embeddings.append(embedding)
                texts.append(content)
                
                metadata = {
                    'source': 'legacy_learning',
                    'chunk_type': 'learning_rec',
                    'keywords': ','.join(item.get('keywords', [])),
                    'category': item.get('category', ''),
                    'id': f"learning_rec_{i}",
                    'title': item.get('title', ''),
                    'content_length': len(content)
                }
                metadatas.append(metadata)
                ids.append(f"learning_{i}")
                
                print(f"     임베딩 차원: {len(embedding)}")
            else:
                print(f"     임베딩 실패")
        
        # 배치로 추가
        if embeddings:
            print(f"\n{len(embeddings)}개 문서를 ChromaDB에 추가 중...")
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"추가 완료! 총 문서 수: {collection.count()}")
            
            # 검색 테스트
            print("\n검색 테스트...")
            test_queries = ["노동시장 전망", "청년 고용", "학습 추천"]
            
            for query in test_queries:
                query_embedding = get_embedding(query)
                if query_embedding:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=2
                    )
                    
                    print(f"\n검색어: '{query}'")
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0], 
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        print(f"  {i+1}. 거리: {distance:.3f}")
                        print(f"     타입: {metadata.get('chunk_type', 'N/A')}")
                        print(f"     제목: {metadata.get('title', 'N/A')}")
                        print(f"     내용: {doc[:80]}...")
        else:
            print("추가할 데이터가 없습니다.")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("레거시 데이터 추가 시작...")
    success = add_legacy_data()
    print(f"작업 결과: {'성공' if success else '실패'}")