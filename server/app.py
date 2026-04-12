try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required") from e

try:
    from models import ShuttleAction, ShuttleObservation
    from server.shuttle_environment import ShuttleEnvironment
except ImportError:
    from models import ShuttleAction, ShuttleObservation
    from shuttle_environment import ShuttleEnvironment

app = create_app(
    ShuttleEnvironment,
    ShuttleAction,
    ShuttleObservation,
    env_name="shuttle-routing-env",
    max_concurrent_envs=4,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
