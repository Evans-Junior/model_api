import subprocess

MEDITRON_MODEL_NAME = "meditron"

def run_meditron_health_assistant(prompt):
    try:
        result = subprocess.run(
            f'echo "{prompt}" | ollama run {MEDITRON_MODEL_NAME}',
            shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error running Meditron model: {str(e)}"
