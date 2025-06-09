# run.py

import subprocess
import sys
import time
import os

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
    # Using Popen to run the backend in a non-blocking way
    backend_process = subprocess.Popen(backend_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Give the backend a moment to start up
    time.sleep(5) 
    
    print("Starting Streamlit frontend...")
    # Run the frontend
    frontend_process = subprocess.Popen(frontend_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Monitor the processes
        while True:
            # You can add logic here to check if processes are alive
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
        # Wait for processes to terminate
        backend_process.wait()
        frontend_process.wait()
        print("Shutdown complete.")


if __name__ == "__main__":
    # Check for dlib model. This is a common point of failure.
    try:
        import dlib
        print("dlib is installed.")
    except ImportError:
        print("Error: dlib is not installed. Please install it.")
        sys.exit(1)

    main()
