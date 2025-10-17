
import os
import logging
import shutil
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from fastapi import UploadFile, HTTPException
import PyPDF2
try:
    import magic
except ImportError:
    # Fallback for file type detection
    magic = None
from pathlib import Path

logger = logging.getLogger(__name__)

class FileUtils:
    """
    Comprehensive file utility class for handling PDF uploads, validation, and processing
    """
    
    def __init__(self):
        # Base upload directory
        self.base_upload_dir = "uploads"
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_mime_types = [
            'application/pdf',
            'application/x-pdf'
        ]
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_upload_dir, exist_ok=True)
    
    def validate_pdf_file(self, file: UploadFile) -> Tuple[bool, str]:
        """
        Validate PDF file for type, size, and security
        Returns: (is_valid, error_message)
        """
        try:
            # Check if file is provided
            if not file:
                return False, "No file provided"
            
            # Check filename
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                return False, "File must be a PDF"
            
            # Read file content for MIME type validation
            content = file.file.read(1024)  # Read first 1KB for magic number detection
            file.file.seek(0)  # Reset file pointer
            
            # Validate MIME type using python-magic
            mime = magic.from_buffer(content, mime=True)
            if mime not in self.allowed_mime_types:
                return False, f"Invalid file type: {mime}. Only PDF files are allowed."
            
            # Check file size by reading entire file (for accurate size)
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)  # Reset pointer
            
            if file_size > self.max_file_size:
                return False, f"File size {file_size/1024/1024:.1f}MB exceeds maximum allowed {self.max_file_size/1024/1024:.1f}MB"
            
            if file_size == 0:
                return False, "File is empty"
            
            # Additional PDF-specific validation
            try:
                pdf_reader = PyPDF2.PdfReader(file.file)
                if len(pdf_reader.pages) == 0:
                    return False, "PDF contains no pages"
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    return False, "Encrypted PDFs are not supported"
                    
            except Exception as e:
                return False, f"Invalid PDF file: {str(e)}"
            
            return True, "File is valid"
            
        except Exception as e:
            logger.error(f"Error validating PDF file: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def generate_unique_filename(self, original_filename: str, client_id: int) -> str:
        """
        Generate a unique filename for storing uploaded files
        """
        try:
            # Extract extension
            extension = Path(original_filename).suffix.lower()
            if not extension:
                extension = '.pdf'
            
            # Generate unique ID
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Clean original filename
            clean_name = Path(original_filename).stem.lower()
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_name = clean_name.replace(' ', '_')
            
            filename = f"client_{client_id}_{clean_name}_{timestamp}_{unique_id}{extension}"
            return filename
            
        except Exception as e:
            logger.error(f"Error generating unique filename: {str(e)}")
            # Fallback filename
            return f"client_{client_id}_{uuid.uuid4().hex[:12]}.pdf"
    
    def get_client_upload_dir(self, client_id: int) -> str:
        """
        Get or create client-specific upload directory
        """
        client_dir = os.path.join(self.base_upload_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        return client_dir
    
    async def save_uploaded_file(self, file: UploadFile, client_id: int) -> Dict[str, Any]:
        """
        Save uploaded file to client-specific directory
        Returns file info dictionary
        """
        try:
            # Validate file first
            is_valid, error_msg = self.validate_pdf_file(file)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(file.filename, client_id)
            client_dir = self.get_client_upload_dir(client_id)
            file_path = os.path.join(client_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                # Read file in chunks to handle large files
                while True:
                    chunk = await file.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    buffer.write(chunk)
            
            # Get file stats
            file_size = os.path.getsize(file_path)
            
            # Extract basic PDF info
            pdf_info = self.extract_pdf_info(file_path)
            
            file_info = {
                "original_filename": file.filename,
                "saved_filename": unique_filename,
                "file_path": file_path,
                "file_size": file_size,
                "client_id": client_id,
                "uploaded_at": datetime.now(),
                "page_count": pdf_info.get("page_count", 0),
                "is_valid": pdf_info.get("is_valid", False),
                "file_type": "pdf"
            }
            
            logger.info(f"File saved successfully: {file_info}")
            return file_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    def extract_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract basic information from PDF file
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = {
                    "page_count": len(pdf_reader.pages),
                    "is_valid": True,
                    "has_text": False,
                    "author": pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                    "title": pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                    "creator": pdf_reader.metadata.get('/Creator', '') if pdf_reader.metadata else ''
                }
                
                # Check if PDF contains extractable text (sample first page)
                if info["page_count"] > 0:
                    try:
                        first_page = pdf_reader.pages[0]
                        text = first_page.extract_text()
                        info["has_text"] = len(text.strip()) > 0
                    except:
                        info["has_text"] = False
                
                return info
                
        except Exception as e:
            logger.error(f"Error extracting PDF info: {str(e)}")
            return {"page_count": 0, "is_valid": False, "has_text": False}
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, int]:
        """
        Extract all text from PDF file
        Returns: (extracted_text, character_count)
        """
        try:
            full_text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            char_count = len(full_text)
            logger.info(f"Extracted {char_count} characters from PDF")
            return full_text, char_count
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return "", 0
    
    def get_client_files(self, client_id: int) -> List[Dict[str, Any]]:
        """
        Get list of all files for a client
        """
        try:
            client_dir = self.get_client_upload_dir(client_id)
            files_info = []
            
            if not os.path.exists(client_dir):
                return files_info
            
            for filename in os.listdir(client_dir):
                file_path = os.path.join(client_dir, filename)
                if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
                    
                    # Get file stats
                    stat = os.stat(file_path)
                    file_info = {
                        "filename": filename,
                        "original_filename": self.get_original_filename(filename),
                        "file_path": file_path,
                        "file_size": stat.st_size,
                        "uploaded_at": datetime.fromtimestamp(stat.st_mtime),
                        "client_id": client_id
                    }
                    
                    # Add PDF info
                    pdf_info = self.extract_pdf_info(file_path)
                    file_info.update(pdf_info)
                    
                    files_info.append(file_info)
            
            # Sort by upload time (newest first)
            files_info.sort(key=lambda x: x["uploaded_at"], reverse=True)
            return files_info
            
        except Exception as e:
            logger.error(f"Error getting client files: {str(e)}")
            return []
    
    def get_original_filename(self, saved_filename: str) -> str:
        """
        Extract original filename from saved filename
        """
        try:
            # Remove client ID, timestamp, and UUID
            parts = saved_filename.split('_')
            if len(parts) > 2:
                # Join parts that belong to original name
                original_parts = parts[2:-2]  # Remove client_id and timestamp/uuid parts
                original_name = '_'.join(original_parts)
                return original_name + '.pdf'
            return saved_filename
        except:
            return saved_filename
    
    def delete_client_file(self, client_id: int, filename: str) -> bool:
        """
        Delete a specific file for a client
        """
        try:
            client_dir = self.get_client_upload_dir(client_id)
            file_path = os.path.join(client_dir, filename)
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting client file: {str(e)}")
            return False
    
    def delete_all_client_files(self, client_id: int) -> bool:
        """
        Delete all files for a client
        """
        try:
            client_dir = self.get_client_upload_dir(client_id)
            
            if os.path.exists(client_dir) and os.path.isdir(client_dir):
                shutil.rmtree(client_dir)
                logger.info(f"Deleted all files for client {client_id}")
                return True
            return True  # Directory doesn't exist, consider it success
            
        except Exception as e:
            logger.error(f"Error deleting all client files: {str(e)}")
            return False
    
    def get_file_stats(self, client_id: int) -> Dict[str, Any]:
        """
        Get statistics about client's files
        """
        try:
            files = self.get_client_files(client_id)
            total_files = len(files)
            total_size = sum(f["file_size"] for f in files)
            total_pages = sum(f.get("page_count", 0) for f in files)
            files_with_text = sum(1 for f in files if f.get("has_text", False))
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_pages": total_pages,
                "files_with_text": files_with_text,
                "files_without_text": total_files - files_with_text,
                "average_file_size_mb": (total_size / (1024 * 1024)) / total_files if total_files > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting file stats: {str(e)}")
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "total_pages": 0,
                "files_with_text": 0,
                "files_without_text": 0,
                "average_file_size_mb": 0
            }
    
    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Clean up files older than specified days
        Returns number of files deleted
        """
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for client_dir_name in os.listdir(self.base_upload_dir):
                client_dir = os.path.join(self.base_upload_dir, client_dir_name)
                if os.path.isdir(client_dir):
                    for filename in os.listdir(client_dir):
                        file_path = os.path.join(client_dir, filename)
                        if os.path.isfile(file_path):
                            file_time = os.path.getmtime(file_path)
                            if file_time < cutoff_time:
                                os.remove(file_path)
                                deleted_count += 1
                                logger.info(f"Cleaned up old file: {file_path}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return 0
    
    def validate_file_path(self, file_path: str) -> bool:
        """
        Validate that file path is within allowed directory and exists
        """
        try:
            # Check if path is within base upload directory
            absolute_path = os.path.abspath(file_path)
            base_absolute = os.path.abspath(self.base_upload_dir)
            
            if not absolute_path.startswith(base_absolute):
                return False
            
            # Check if file exists and is a file
            return os.path.exists(absolute_path) and os.path.isfile(absolute_path)
            
        except Exception:
            return False
    
    def get_file_content(self, file_path: str) -> Optional[bytes]:
        """
        Safely read file content
        """
        try:
            if not self.validate_file_path(file_path):
                return None
            
            with open(file_path, 'rb') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            return None


# Create global instance
file_utils = FileUtils()


# Utility functions
def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def is_safe_filename(filename: str) -> bool:
    """Check if filename is safe to use"""
    if not filename or len(filename) > 255:
        return False
    
    # Check for path traversal attempts
    dangerous_patterns = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    return not any(pattern in filename for pattern in dangerous_patterns)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing dangerous characters"""
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    return filename


# Example usage and testing
if __name__ == "__main__":
    # Test the utility functions
    print("FileUtils initialized successfully!")
    print(f"Base upload directory: {file_utils.base_upload_dir}")
    print(f"Max file size: {format_file_size(file_utils.max_file_size)}")
    print("Allowed MIME types:", file_utils.allowed_mime_types)
    
    # Test filename generation
    test_filename = file_utils.generate_unique_filename("test document.pdf", 123)
    print(f"Generated filename: {test_filename}")
    
    # Test file size formatting
    print(f"1MB in bytes: {format_file_size(1024*1024)}")
    print(f"2.5MB in bytes: {format_file_size(2.5*1024*1024)}")
