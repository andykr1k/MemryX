import subprocess
import sys
import time

def main():
    """
    This script starts the FastAPI backend and the Streamlit frontend.
    It ensures that both processes are running concurrently.
    """
    # Command to run FastAPI backend with uvicorn
    backend_command = [
        sys.executable, "-m", "uvicorn", "Backend:app", "--host", "127.0.0.1", "--port", "8000"
    ]

    # Command to run Streamlit frontend
    frontend_command = [
        sys.executable, "-m", "streamlit", "run", "Frontend.py"
    ]
    
    print("Starting FastAPI backend...")
    backend_process = subprocess.Popen(backend_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    time.sleep(5)  # Give the backend time to spin up

    print("Starting Streamlit frontend...")
    frontend_process = subprocess.Popen(frontend_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Monitor the processes
        while True:
            if backend_process.poll() is not None:
                print("Backend process has terminated.")
                break
            if frontend_process.poll() is not None:
                print("Frontend process has terminated.")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    # Optional: Check for critical packages
    # try:
    #     import insightface
    #     import onnxruntime
    #     print("InsightFace and ONNX Runtime are installed.")
    # except ImportError as e:
    #     print(f"Missing dependency: {e.name}. Please install it.")
    #     sys.exit(1)

    main()
