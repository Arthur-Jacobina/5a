import dspy
from mem0 import Memory

from .memory import MemoryTools
from .utils import get_current_time
from .observability import log_mlflow

# Optional MLflow integration
try:
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class MemoryQA(dspy.Signature):
    """
    You're a helpful assistant and have access to memory method.
    Whenever you answer a user's input, remember to store the information in memory
    so that you can use it later.
    """
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

class MemoryReActAgent(dspy.Module):
    """A ReAct agent enhanced with Mem0 memory capabilities."""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory_tools = MemoryTools(memory)

        # Create tools list for ReAct
        self.tools = [
            self.memory_tools.store_memory,
            self.memory_tools.search_memories,
            self.memory_tools.get_all_memories,
            get_current_time,
            self.set_reminder,
            self.get_preferences,
            self.update_preferences,
        ]

        # Initialize ReAct with our tools
        self.react = dspy.ReAct(
            signature=MemoryQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """Process user input with memory-aware reasoning."""
        return self.react(user_input=user_input)

    def log_to_mlflow(self, run_name: str = None, experiment_name: str = "default"):
        """Log the DSPy model to MLflow for versioning and deployment."""
        if not MLFLOW_AVAILABLE:
            print("⚠️  MLflow not available. Skipping model logging.")
            return None
        return log_mlflow(self, run_name=run_name, experiment_name=experiment_name)

    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
        """Set a reminder for the user."""
        reminder = f"Reminder set for {date_time}: {reminder_text}"
        return self.memory_tools.store_memory(
            f"REMINDER: {reminder}", 
            user_id=user_id
        )

    def get_preferences(self, category: str = "general", user_id: str = "default_user") -> str:
        """Get user preferences for a specific category."""
        query = f"user preferences {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )

    def update_preferences(self, category: str, preference: str, user_id: str = "default_user") -> str:
        """Update user preferences."""
        preference_text = f"User preference for {category}: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )
    