import os
import mlflow

def setup_mlflow():
    """Setup MLflow with local file-based tracking"""
    try:
        # Use local file-based tracking instead of remote server
        mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlflow_tracking_dir, exist_ok=True)
        
        # Set tracking URI to local directory
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
        
        # Set experiment
        mlflow.set_experiment("DSPy-Memory-Agent")
        
        # Enable autologging
        mlflow.dspy.autolog()
        
        print(f"✅ MLflow initialized with local tracking at: {mlflow_tracking_dir}")
        
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {e}")
        print("🔄 Continuing without MLflow tracking...")

def log_mlflow(agent, run_name: str = None, experiment_name: str = "default"):
    """Log the DSPy agent model to MLflow for versioning and deployment."""
    try:
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            model_info = mlflow.dspy.log_model(
                agent,
                name="memory_react_agent",
                input_example="What did we discuss about project management?",
            )
            
            mlflow.log_param("max_iters", 6)
            mlflow.log_param("num_tools", len(agent.tools))
            mlflow.log_param("tool_names", [tool.__name__ for tool in agent.tools])
            
            print(f"✅ Model logged to MLflow successfully!")
            return model_info
            
    except Exception as e:
        print(f"⚠️  Error logging to MLflow: {e}")
        return None
