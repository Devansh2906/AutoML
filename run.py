import os
import subprocess

if __name__ == "__main__":
    # Get the absolute path to the frontend/app.py script
    app_path = os.path.join("frontend", "app.py")
    
    # Launch Streamlit server automatically
    print("Launching the AutoML Platform local web server...")
    subprocess.run(["streamlit", "run", app_path])