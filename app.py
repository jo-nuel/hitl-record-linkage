from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"), run_name="__main__")
