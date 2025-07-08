from abc import ABC, abstractmethod

class MemoryTools(ABC):
    @abstractmethod
    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        pass

    @abstractmethod
    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        pass
    
    @abstractmethod
    def get_all_memories(self, user_id: str = "default_user") -> str:
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, new_content: str) -> str:
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> str:
        pass