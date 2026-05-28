"""
기존 실험 폴더에 흩어진 report.json 파일들을 읽어
dashboard/reports.json 단일 스토어로 만들어 Drive에 업로드합니다.

Usage:
  python build_reports_store.py            # dry-run
  python build_reports_store.py --execute  # 실제 업로드
"""
import json, sys, warnings
warnings.filterwarnings("ignore")

import requests
from google.oauth2 import service_account
from google.auth.transport import requests as ga_requests

PROJECT_FOLDER_ID = "1kGw6h5K38jBHI-UR1r5EClyZWb6jpc6B"
SA_INFO = {
    "type": "service_account",
    "project_id": "graph-aml",
    "private_key_id": "94834f9c427ad3a43525d9faf2d917e2e65dfbb5",
    "private_key": (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCd1cFhkNTr6xg3\n"
        "LCPfZvfjT7dmypreoKxrvxwWmIb8TFQTeqczv26Yth1btQvEJJfs+1NsXGOPrhre\n"
        "BY25M267clEC0Ps7oI72Q4iuoZSUYDp0wNUSHkkYFnFqNWpT93Qcr3Rju5rLTX+y\n"
        "0/ujLCOun5+x4qHzA0l3CbL1BkeE5HLJeDolMaaREz8gc4dM2xsnJFK7DxAGeFaD\n"
        "GHcicP/IBvSrWE05TCNECfm7fswwd9FAIjYCo//hb2LvAbrOmgrQr/sBxqcNrbWt\n"
        "tVrcYT2ElScgbaznBIXZlduSeZm2MWTeXj+3rITTsP9AzKOZQQeqC72FYWtoniUh\n"
        "2yaE0OlfAgMBAAECggEAA01ew5GkqspF55hFqIhpJPkyKfU7jUSI/E4eIn45dhuL\n"
        "YhxIVZr/5N5hOd3NVmK6R6vbIHgZtdQGMuFPMvHtCKtJH552Cywo9uUN0zLalB95\n"
        "SZ+5NYJVtCp6tVFTHVj0NcoZTMw23mG2Nhcc+4UnuqETVcTwmiWA9XvJ9zoU5/uu\n"
        "h/0Y0ihr+Gvki67nn+cK5UY78PV0K3IpdFQCHwuG2R5icEx72ZNkJFC2oIl5MeVo\n"
        "1Dv1/CBF2qJ5t/20HTjDgljDoZX1fEJUVsUR/IGCWcRVVvfJ1TFjUWaLaU1sy0o3\n"
        "8cu4PMVFhmfb3g6K4vKp0m1tfG4gfkKCVPoGg04B3QKBgQDPzbHIxIqNF4ZW96Dj\n"
        "URxxWzSUNZlnKoXhCFyZxVJR07mhreX23F2L2bSggytulcmeSp7CqaKpjuTVqhOo\n"
        "SwstrmqMG6BZXFOuhiuR4Kk87dMe9XRJdCxRWOwnBSahpoEplo6oXw4gAIwYWiwx\n"
        "T5C5szrFUhk80YWj9L6BpnEsWwKBgQDCcS/fNEv2/sm/MfsBDzFeB5hD2l3UwMBJ\n"
        "I7TXeQETdkv5tkvhSh+XYHcXFKWQkTOOSrhW+9rZX6RoB0U+H3I5YBS1QGdDOlUX\n"
        "2ISSY8ZnuLQSUWkyuQjWyVDWxzt9G9pR/kgh6Yd3tfpgTePTUAb6Ng/Pnn1/84tH\n"
        "PSnu1aBWTQKBgQCJjgmfcqqcVvQwYV742lpPlyYo7YoMRpO0sIpLp9ikHdkFc02E\n"
        "qb6qsoPktK9tVm3OAGszRINOZi6IWTsF7hcKOCiDck4kmP8zydDRkbu1f2B/X8+I\n"
        "SASGHKzF75zw6H0bgHQSdEmvWW1jOV2Djr2oj0HaGExoe/FQ5NOukvTbfQKBgGDr\n"
        "gFw2uiLMz40xAZd+ljHzgS9ZOmohBfevB6Zb13B3B9nZxyruAp8240Wq8fgEmHk1\n"
        "v3sEIQs3BEEiVp5nmE0HGmtaRd6Zxe6T60j42N28kG2NDO3Ok5xUTqowNvPenU0/\n"
        "fX8B45eFKt80E/qxqjiwF+N6cb4EjIke8Lbu3vQFAoGBAIB00g4oPIZoxAgLHYDZ\n"
        "JlPdIF1kg+0pWNpR7Jb2zgDMgu8ShcmgZsrNKWxFLJt+JEBox4zVvSkZEpTDz2+E\n"
        "wLuA2KboOUFG8VQz5NlBqRiVZN3553oCyAnPv5FzdyvomgH3WSjkXsIzMzsrwW9B\n"
        "nUUF36k7rWM0n5yVQ9LRbjeU\n"
        "-----END PRIVATE KEY-----\n"
    ),
    "client_email": "graph-aml-dashboard@graph-aml.iam.gserviceaccount.com",
    "client_id": "101383650744755843059",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/graph-aml-dashboard%40graph-aml.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com",
}
EXECUTE = "--execute" in sys.argv

def get_token():
    creds = service_account.Credentials.from_service_account_info(
        SA_INFO, scopes=["https://www.googleapis.com/auth/drive"]
    )
    creds.refresh(ga_requests.Request())
    return creds.token

def find_folder(name, parent_id, token):
    q = (f"'{parent_id}' in parents and name='{name}'"
         " and mimeType='application/vnd.google-apps.folder' and trashed=false")
    r = requests.get("https://www.googleapis.com/drive/v3/files",
                     headers={"Authorization": f"Bearer {token}"},
                     params={"q": q, "fields": "files(id)", "pageSize": 1}, timeout=15)
    files = r.json().get("files", []) if r.ok else []
    return files[0]["id"] if files else ""

def find_file(name, parent_id, token):
    q = f"'{parent_id}' in parents and name='{name}' and trashed=false"
    r = requests.get("https://www.googleapis.com/drive/v3/files",
                     headers={"Authorization": f"Bearer {token}"},
                     params={"q": q, "fields": "files(id)", "pageSize": 1}, timeout=15)
    files = r.json().get("files", []) if r.ok else []
    return files[0]["id"] if files else ""

def read_json_file(file_id, token):
    r = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                     headers={"Authorization": f"Bearer {token}"},
                     params={"alt": "media"}, timeout=15)
    try:
        return r.json() if r.ok else None
    except Exception:
        return None

def patch_file(file_id, data, token):
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    r = requests.patch(
        f"https://www.googleapis.com/upload/drive/v3/files/{file_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        params={"uploadType": "media"}, data=payload, timeout=15,
    )
    if not r.ok:
        print(f"  !! patch failed {r.status_code}: {r.text[:200]}")
    return r.ok

# ── Sources: (store_key, folder_id_getter) ─────────────────────────────────
print("Getting token...")
TOKEN = get_token()
print("OK\n")

# 1. 기존 실험 폴더의 report.json 들을 읽어 store 딕셔너리 구성
store = {}

SOURCES = [
    # (tab_name, exp_name, path_to_folder_fn)
    # GNN: gnn/GNN-XX/report.json
    ("GNN Result", None, lambda: find_folder("gnn", PROJECT_FOLDER_ID, TOKEN)),
    # ML:  ml/ml-XX/report.json
    ("ML Result", None, lambda: find_folder("ml", PROJECT_FOLDER_ID, TOKEN)),
    # WOE: data/ml/woe_iv/ml-XX/report.json
    ("Univariate Analysis", None, lambda: (
        lambda data_id: (
            lambda data_ml_id: find_folder("woe_iv", data_ml_id, TOKEN)
            if data_ml_id else ""
        )(find_folder("ml", data_id, TOKEN))
        if data_id else ""
    )(find_folder("data", PROJECT_FOLDER_ID, TOKEN))),
]

TAB_ROOTS = {
    "GNN Result":          find_folder("gnn", PROJECT_FOLDER_ID, TOKEN),
    "ML Result":           find_folder("ml",  PROJECT_FOLDER_ID, TOKEN),
    "Univariate Analysis": (lambda d: (lambda m: find_folder("woe_iv", m, TOKEN) if m else "")(
                                find_folder("ml", d, TOKEN)) if d else "")(
                                find_folder("data", PROJECT_FOLDER_ID, TOKEN)),
}

print("Scanning experiment folders for existing report.json files...\n")
for tab_name, root_id in TAB_ROOTS.items():
    if not root_id:
        print(f"  [skip] {tab_name} — root folder not found")
        continue
    # list subfolders of root
    q = f"'{root_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    r = requests.get("https://www.googleapis.com/drive/v3/files",
                     headers={"Authorization": f"Bearer {TOKEN}"},
                     params={"q": q, "fields": "files(id,name)", "pageSize": 100}, timeout=15)
    exp_folders = r.json().get("files", []) if r.ok else []
    for ef in exp_folders:
        fid = find_file("report.json", ef["id"], TOKEN)
        if not fid:
            continue
        data = read_json_file(fid, TOKEN)
        if not data:
            continue
        key = f"{tab_name}__{ef['name']}"
        store[key] = data
        print(f"  [found] {key}")

print(f"\nTotal entries collected: {len(store)}")
if not store:
    print("Nothing to migrate. Store will be empty ({}).")

# 2. dashboard/reports.json 찾기 또는 확인
dash_id = find_folder("dashboard", PROJECT_FOLDER_ID, TOKEN)
if not dash_id:
    print("\nERROR: 'dashboard' folder not found in PROJECT_FOLDER_ID.")
    sys.exit(1)

store_fid = find_file("reports.json", dash_id, TOKEN)
if not store_fid:
    print("\nERROR: reports.json not found in dashboard/.")
    print("Drive에서 dashboard 폴더 안에 내용이 {} 인 reports.json 파일을 먼저 만들어 주세요.")
    sys.exit(1)

print(f"\ndashboard/reports.json found: {store_fid}")

if EXECUTE:
    ok = patch_file(store_fid, store, TOKEN)
    print("Upload OK" if ok else "Upload FAILED")
else:
    print("\n[dry-run] 위 내용을 dashboard/reports.json에 쓰려면 --execute 옵션을 붙이세요.")
    print(json.dumps(store, ensure_ascii=False, indent=2)[:500] + ("..." if len(json.dumps(store)) > 500 else ""))
