# app/document_processor.py
"""
문서 처리 및 청킹 시스템
PDF 파일을 청크 단위로 분할하여 벡터 데이터베이스에 저장 준비
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
import PyPDF2
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        문서 처리기 초기화
        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 중복 크기 (문자 수)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        PDF에서 텍스트 추출
        Args:
            pdf_path: PDF 파일 경로
        Returns:
            List[Dict]: 페이지별 텍스트 정보
        """
        pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # 빈 페이지 제외
                            pages.append({
                                'page_number': page_num,
                                'text': self._clean_text(text),
                                'char_count': len(text),
                                'source': os.path.basename(pdf_path)
                            })
                    except Exception as e:
                        logger.warning(f"페이지 {page_num} 텍스트 추출 실패: {e}")
                        continue
                        
            logger.info(f"PDF 추출 완료: {len(pages)}페이지, {pdf_path}")
            return pages
            
        except Exception as e:
            logger.error(f"PDF 파일 읽기 실패: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 여러 공백을 하나로 합치기
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 문자 제거
        text = re.sub(r'[^\w\s\-\.\,\(\)\[\]\:\;\%\+\=\<\>\'\"\n가-힣]', '', text)
        
        # 연속된 줄바꿈 정리
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def create_chunks_from_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        페이지 데이터를 청크로 분할
        Args:
            pages: 페이지별 텍스트 정보
        Returns:
            List[Dict]: 청크 정보
        """
        all_chunks = []
        
        for page_info in pages:
            page_text = page_info['text']
            page_num = page_info['page_number']
            source = page_info['source']
            
            # 페이지가 청크 크기보다 작으면 통째로 처리
            if len(page_text) <= self.chunk_size:
                chunk = self._create_chunk(
                    content=page_text,
                    source=source,
                    page_number=page_num,
                    chunk_index=0,
                    chunk_type='page'
                )
                all_chunks.append(chunk)
            else:
                # 큰 페이지는 여러 청크로 분할
                page_chunks = self._split_text_into_chunks(page_text)
                
                for i, chunk_text in enumerate(page_chunks):
                    chunk = self._create_chunk(
                        content=chunk_text,
                        source=source,
                        page_number=page_num,
                        chunk_index=i,
                        chunk_type='chunk'
                    )
                    all_chunks.append(chunk)
        
        logger.info(f"청킹 완료: {len(all_chunks)}개 청크 생성")
        return all_chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 마지막 청크가 아니면 문장 경계에서 자르기
            if end < len(text):
                # 문장 끝 찾기 (마침표, 느낌표, 물음표 후)
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            # 최소 길이 필터링 (50자 미만 청크 제외)
            if chunk_text and len(chunk_text) >= 50:
                chunks.append(chunk_text)
            elif chunk_text and len(chunk_text) < 50:
                # 매우 짧은 청크는 다음 청크와 합치거나 무시
                logger.debug(f"짧은 청크 필터링됨: '{chunk_text[:30]}...' (길이: {len(chunk_text)})")
            
            # 다음 청크 시작점 (중복 고려)
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    def _create_chunk(self, content: str, source: str, page_number: int, 
                     chunk_index: int, chunk_type: str) -> Dict:
        """청크 정보 생성"""
        # 키워드 추출 (간단한 방식)
        keywords = self._extract_keywords(content)
        
        return {
            'content': content,
            'source': source,
            'page': page_number,
            'chunk_index': chunk_index,
            'chunk_type': chunk_type,
            'keywords': keywords,
            'content_length': len(content),
            'processed_at': datetime.now().isoformat()
        }
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """텍스트에서 키워드 추출 (간단한 방식)"""
        # 노동시장 관련 주요 키워드들
        labor_keywords = [
            '취업자', '실업자', '실업률', '고용률', '노동시장', '청년층', '고령층',
            '제조업', '건설업', '서비스업', '정보통신업', '보건업', '사회복지',
            '상용직', '임시직', '일용직', '비정규직', '정규직',
            '임금', '급여', '소득', '연봉', '최저임금',
            '경제활동인구', '비경제활동인구', '생산가능인구',
            '고용정책', '일자리', '채용', '구직', '취업', '전망'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        # 텍스트에서 키워드 찾기
        for keyword in labor_keywords:
            if keyword.lower() in text_lower or keyword in text:
                found_keywords.append(keyword)
        
        # 숫자와 함께 나오는 패턴 찾기 (예: "18.4만 명", "2024년")
        number_patterns = re.findall(r'(\d+[\.\,]?\d*[만천억]?\s*[명원개년월%분기])', text)
        for pattern in number_patterns[:5]:  # 최대 5개
            found_keywords.append(pattern.strip())
        
        return found_keywords[:max_keywords]
    
    def process_structured_content(self, pages: List[Dict]) -> List[Dict]:
        """
        구조화된 내용 처리 (표, 섹션별 분리)
        Args:
            pages: 페이지 데이터
        Returns:
            List[Dict]: 구조화된 청크들
        """
        structured_chunks = []
        
        for page_info in pages:
            text = page_info['text']
            page_num = page_info['page_number']
            source = page_info['source']
            
            # 섹션 제목 찾기
            sections = self._identify_sections(text)
            
            if sections:
                for section in sections:
                    chunk = self._create_chunk(
                        content=section['content'],
                        source=source,
                        page_number=page_num,
                        chunk_index=section['index'],
                        chunk_type='section'
                    )
                    chunk['section_title'] = section.get('title', '')
                    structured_chunks.append(chunk)
            
            # 표 데이터 추출
            tables = self._extract_table_like_content(text)
            for i, table in enumerate(tables):
                chunk = self._create_chunk(
                    content=table,
                    source=source,
                    page_number=page_num,
                    chunk_index=i,
                    chunk_type='table'
                )
                structured_chunks.append(chunk)
        
        logger.info(f"구조화된 콘텐츠 처리 완료: {len(structured_chunks)}개")
        return structured_chunks
    
    def _identify_sections(self, text: str) -> List[Dict]:
        """섹션 식별"""
        sections = []
        
        # 섹션 패턴 (예: "1. ", "가. ", "Ⅰ. " 등)
        section_pattern = r'^\s*([IⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩivx]+\.|\d+\.|\w+\.)(.+?)(?=^\s*([IⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩivx]+\.|\d+\.|\w+\.)|$)'
        
        matches = re.finditer(section_pattern, text, re.MULTILINE | re.DOTALL)
        
        for i, match in enumerate(matches):
            sections.append({
                'index': i,
                'title': match.group(2).strip()[:100],  # 제목 길이 제한
                'content': match.group(0).strip()
            })
        
        return sections
    
    def _extract_table_like_content(self, text: str) -> List[str]:
        """표 형식 내용 추출"""
        tables = []
        
        # 표 패턴 찾기 (여러 줄에 걸쳐 숫자와 텍스트가 규칙적으로 나타나는 경우)
        lines = text.split('\n')
        current_table = []
        
        for line in lines:
            line = line.strip()
            
            # 표 같은 라인 판별 (탭이나 여러 공백으로 구분된 데이터)
            if self._is_table_line(line):
                current_table.append(line)
            else:
                if len(current_table) >= 3:  # 최소 3줄 이상
                    tables.append('\n'.join(current_table))
                current_table = []
        
        # 마지막 표 처리
        if len(current_table) >= 3:
            tables.append('\n'.join(current_table))
        
        return tables
    
    def _is_table_line(self, line: str) -> bool:
        """라인이 표 데이터인지 판별"""
        if not line or len(line) < 10:
            return False
        
        # 숫자와 구분자가 있는 패턴
        number_count = len(re.findall(r'\d+[\.\,]?\d*', line))
        separator_count = len(re.findall(r'\s{2,}|\t', line))
        
        return number_count >= 2 and separator_count >= 1
    
    def process_pdf_file(self, pdf_path: str, use_structured: bool = True) -> List[Dict]:
        """
        PDF 파일 전체 처리 파이프라인
        Args:
            pdf_path: PDF 파일 경로
            use_structured: 구조화 처리 사용 여부
        Returns:
            List[Dict]: 처리된 청크들
        """
        logger.info(f"PDF 처리 시작: {pdf_path}")
        
        # 1. PDF 텍스트 추출
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            logger.error("PDF에서 텍스트를 추출할 수 없습니다.")
            return []
        
        # 2. 기본 청킹
        basic_chunks = self.create_chunks_from_pages(pages)
        
        # 3. 구조화된 처리 (옵션)
        if use_structured:
            structured_chunks = self.process_structured_content(pages)
            # 중복 제거하면서 병합
            all_chunks = basic_chunks + structured_chunks
        else:
            all_chunks = basic_chunks
        
        logger.info(f"PDF 처리 완료: {len(all_chunks)}개 청크 생성")
        return all_chunks