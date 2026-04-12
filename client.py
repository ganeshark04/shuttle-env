from openenv import SyncEnvClient

def get_client(base_url: str):
    return SyncEnvClient(base_url=base_url)
