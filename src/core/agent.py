import dspy
from mem0 import Memory
from typing import List, Optional

from .memory.memory import MemoryTools
from .utils import get_current_time
from .observability.mlflow import log_mlflow
    
class Agent(dspy.Module):
    """A generic agent that can be constructed with various plugins like Memory, Tools, Signatures, etc."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        memory: Optional[Memory] = None,
        tools: Optional[List[callable]] = None,
        signature: Optional[dspy.Signature] = None,
        reasoning_system: str = "react",
        observability: bool = False,
        max_iters: int = 6,
        **kwargs
    ):
        """
        Initialize the Agent with configurable plugins.
        
        Args:
            model: Model to use for the agent
            memory: Memory instance for storing and retrieving memories
            tools: List of tools/functions the agent can use
            signature: DSPy signature for the agent's input/output format
            reasoning_system: Type of reasoning system ("react", "cot", "basic")
            observability: Whether to enable observability features
            max_iters: Maximum iterations for reasoning systems
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        
        self.observability = observability
        self.max_iters = max_iters
        
        self.memory_tools = memory and MemoryTools(memory)
        
        # Setup tools
        self.tools = self._setup_tools(tools)
        
        # Setup signature
        self.signature = signature
        
        # Setup reasoning system
        self.reasoning_system = self._setup_reasoning_system(reasoning_system)
        
        # Store additional configuration
        self.config = kwargs

        # Configure DSPy with model - default to gpt-4o-mini
        dspy.configure(lm=dspy.LM(model))

    def _setup_tools(self, custom_tools: Optional[List[callable]] = None) -> List[callable]:
        """ Setup the tools list with memory tools and custom tools.
        
        Args:
            custom_tools: List of custom tools to add to the agent
            
        Returns:
            List of tools to use in the agent
        """

        # Initialize tools with current time utility (essential for all agents)
        tools = [get_current_time]
        
        # Add memory if provided
        if self.memory_tools:
            tools.extend([
                self.memory_tools.store_memory,
                self.memory_tools.search_memories,
                self.memory_tools.get_all_memories,
                self.memory_tools.update_memory,
                self.memory_tools.delete_memory,
                self.set_reminder,
                self.get_preferences,
                self.update_preferences,
            ])
        
        # Add tools if provided
        if custom_tools:
            tools.extend(custom_tools)
        
        return tools
    
    def _setup_reasoning_system(self, reasoning_system: str):
        """ Setup the reasoning system based on the specified type.
        
        Args:
            reasoning_system: Type of reasoning system ("react", "cot", "basic")
            
        Returns:
            Reasoning system instance
        """
        match reasoning_system.lower():
            # React Agent with tool calling capabilities (default)
            case "react":
                return dspy.ReAct(
                    signature=self.signature,
                    tools=self.tools,
                    max_iters=self.max_iters
                )
            # Chain of Thought agent
            case "cot":
                return dspy.ChainOfThought(self.signature)
            # Basic agent
            case "basic":
                return dspy.Predict(self.signature)
            case _:
                raise ValueError(f"Unsupported reasoning system: {reasoning_system}")

    def forward(self, user_input: str, **kwargs):
        """Process user input through the reasoning system."""
        if hasattr(self.reasoning_system, '__call__'):
            return self.reasoning_system(user_input=user_input, **kwargs)
        else:
            return self.reasoning_system(user_input, **kwargs)
    
    def log_to_mlflow(self, run_name: str = None, experiment_name: str = "default"):
        """Log the DSPy model to MLflow for versioning and deployment."""
        if self.observability:
            return log_mlflow(self, run_name=run_name, experiment_name=experiment_name)
        else:
            raise RuntimeError("Observability is not enabled for this agent")
    
    # Memory-related methods (only available if memory is configured) 
    # TODO: Put in a separate file
    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = None) -> str:
        """Set a reminder for the user."""
        if not self.memory_tools:
            raise RuntimeError("Memory is not configured for this agent")
        
        user_id = user_id or "default_user"
        reminder = f"Reminder set for {date_time}: {reminder_text}"
        return self.memory_tools.store_memory(
            f"REMINDER: {reminder}", 
            user_id=user_id
        )
    
    def get_preferences(self, category: str = "general", user_id: str = None) -> str:
        """Get user preferences for a specific category."""
        if not self.memory_tools:
            raise RuntimeError("Memory is not configured for this agent")
        
        user_id = user_id or "default_user"
        query = f"user preferences {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )
    
    def update_preferences(self, category: str, preference: str, user_id: str = None) -> str:
        """Update user preferences."""
        if not self.memory_tools:
            raise RuntimeError("Memory is not configured for this agent")
        
        user_id = user_id or "default_user"
        
        preference_text = f"User preference for {category}: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )
    
    def __call__(self, user_input: str = None, **kwargs) -> str:
        """Call the agent with a user input."""
        return self.forward(user_input, **kwargs)
