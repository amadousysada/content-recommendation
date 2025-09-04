import os, json
import azure.functions as func

app = func.FunctionApp()

# --- Helpers & caches ---
_blob_client = None
_index_cache = None

def _client():
    """Créé le client Blob à la première demande (import paresseux + garde-fous)."""
    global _blob_client
    if _blob_client is None:
        try:
            from azure.storage.blob import BlobServiceClient
        except Exception as e:
            # Message explicite si la lib n'est pas installée
            raise RuntimeError("Dépendance manquante: azure-storage-blob") from e

        conn = os.getenv("AzureWebJobsStorage")
        if not conn:
            raise RuntimeError("App setting 'AzureWebJobsStorage' manquante")
        _blob_client = BlobServiceClient.from_connection_string(conn)
    return _blob_client

def _get_blob_text(container: str, name: str) -> str:
    bc = _client().get_container_client(container).get_blob_client(name)
    return bc.download_blob().content_as_text()

def _upload_json(container: str, name: str, data: dict):
    cc = _client().get_container_client(container)
    cc.upload_blob(name, json.dumps(data, ensure_ascii=False), overwrite=True)

def _get_index() -> dict:
    global _index_cache
    if _index_cache is None:
        container = os.environ.get("BLOB_CONTAINER_MODELS", "models")
        txt = _get_blob_text(container, "index.json")
        _index_cache = json.loads(txt)
    return _index_cache

def _compute_on_demand(user_id: str) -> dict:
    idx = _get_index()
    sugs = idx.get(user_id) or idx.get("_default", [])
    return {"user_id": user_id, "suggestions": sugs[:5], "source": "realtime"}

@app.route(route="ping", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def ping(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("pong", status_code=200)

# --- HTTP: reco ---
@app.route(route="recommend_get", auth_level=func.AuthLevel.ANONYMOUS)
def recommend_get(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.params.get("user_id")
    if not user_id:
        return func.HttpResponse(json.dumps({"error":"missing user_id"}),
                                 status_code=400, mimetype="application/json")

    realtime = (req.params.get("realtime", "false").lower() == "true")
    if not realtime:
        container = os.environ.get("BLOB_CONTAINER_PRECOMPUTED", "precomputed")
        name = f"{user_id}.json"
        bc = _client().get_container_client(container).get_blob_client(name)
        if bc.exists():
            data = json.loads(bc.download_blob().content_as_text())
            return func.HttpResponse(json.dumps(data, ensure_ascii=False),
                                     mimetype="application/json", status_code=200)

    # fallback temps réel
    data = _compute_on_demand(user_id)
    return func.HttpResponse(json.dumps(data, ensure_ascii=False),
                             mimetype="application/json", status_code=200)

# --- HTTP: pré-calcul à la demande ---
@app.route(route="precompute_http", methods=["GET","POST"], auth_level=func.AuthLevel.ANONYMOUS)
def precompute_http(req: func.HttpRequest) -> func.HttpResponse:
    c_models = os.environ.get("BLOB_CONTAINER_MODELS", "models")
    c_data   = os.environ.get("BLOB_CONTAINER_DATA", "data")
    c_prec   = os.environ.get("BLOB_CONTAINER_PRECOMPUTED", "precomputed")

    users = json.loads(_get_blob_text(c_data, "users.json"))
    index = json.loads(_get_blob_text(c_models, "index.json"))

    count = 0
    for uid in users:
        sugs = index.get(str(uid)) or index.get("_default", [])
        out  = {"user_id": str(uid), "suggestions": sugs[:5], "source": "precomputed"}
        _upload_json(c_prec, f"{uid}.json", out)
        count += 1

    return func.HttpResponse(json.dumps({"status":"ok","precomputed":count}, ensure_ascii=False),
                             mimetype="application/json", status_code=200)

# --- TIMER: pré-calcul périodique (toutes les 30s pour tester) ---
@app.timer_trigger(schedule="*/30 * * * * *", arg_name="mytimer")
def precompute_timer(mytimer: func.TimerRequest) -> None:
    c_models = os.environ.get("BLOB_CONTAINER_MODELS", "models")
    c_data   = os.environ.get("BLOB_CONTAINER_DATA", "data")
    c_prec   = os.environ.get("BLOB_CONTAINER_PRECOMPUTED", "precomputed")

    users = json.loads(_get_blob_text(c_data, "users.json"))
    index = json.loads(_get_blob_text(c_models, "index.json"))

    for uid in users:
        sugs = index.get(str(uid)) or index.get("_default", [])
        out  = {"user_id": str(uid), "suggestions": sugs[:5], "source": "precomputed"}
        _upload_json(c_prec, f"{uid}.json", out)

@app.route(route="diag", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def diag(req: func.HttpRequest) -> func.HttpResponse:
    # 1) quel compte lit réellement la Function ?
    conn = os.getenv("AzureWebJobsStorage", "")
    account = None
    for kv in conn.split(";"):
        if kv.startswith("AccountName="):
            account = kv.split("=", 1)[1]
            break

    # 2) est-ce que models/index.json et data/users.json sont *vraiment* lisibles ?
    models = os.getenv("BLOB_CONTAINER_MODELS", "models")
    data   = os.getenv("BLOB_CONTAINER_DATA", "data")
    prec   = os.getenv("BLOB_CONTAINER_PRECOMPUTED", "precomputed")

    ok_index = ok_users = ok_prec_sample = False
    err = None
    try:
        # essaie de lire les deux blobs de base
        from azure.core.exceptions import ResourceNotFoundError
        try:
            _ = _get_blob_text(models, "index.json")
            ok_index = True
        except ResourceNotFoundError:
            ok_index = False

        try:
            _ = _get_blob_text(data, "users.json")
            ok_users = True
        except ResourceNotFoundError:
            ok_users = False

        # existence d’un pré-calcul pour user_id=1 ?
        try:
            from azure.storage.blob import BlobServiceClient
            bc = _client().get_container_client(prec).get_blob_client("1.json")
            ok_prec_sample = bc.exists()
        except Exception:
            ok_prec_sample = False

    except Exception as e:
        err = str(e)

    payload = {
        "account_read_by_function": account,
        "containers": {"models": models, "data": data, "precomputed": prec},
        "can_read": {"models/index.json": ok_index, "data/users.json": ok_users, "precomputed/1.json": ok_prec_sample},
        "error": err,
    }
    return func.HttpResponse(json.dumps(payload), mimetype="application/json", status_code=200)