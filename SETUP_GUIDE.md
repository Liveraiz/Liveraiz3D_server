# SSISO API 환경 설정 및 실행 가이드

이 문서는 **초보자**를 위한 SSISO API 프로젝트의 환경 설정 및 실행 방법을 설명합니다.

---

## 1. 프로젝트 개요

SSISO API는 **의료 영상 분석**을 위한 딥러닝 기반 API 서버입니다.

### 주요 기능

-   **파일 변환**: `.nii.gz` ↔ `.nrrd` 형식 변환
-   **의료 영상 세그멘테이션**: 간, 혈관, 췌장, 신장 등의 CT/MRI 영상 분석
-   **지원 모델**:
    -   HCC-PV (간암 문맥 영역, MRI)
    -   LDLT-PVHV (생체간이식 문맥/간정맥 영역, CT)
    -   Liver-Vein-Spleen (간, 혈관, 비장, CT)
    -   Pancreas-Cancer (췌장암, CT)
    -   Parenchyma-Cancer (실질/암, MRI)
    -   nnUNet 모델들 (Liver 5-section, Kidney)

---

## 2. 사전 준비사항

### 필수 설치 프로그램

#### 2.1 Python 설치

1. [Python 공식 사이트](https://www.python.org/downloads/)에서 **Python 3.11** 다운로드
2. 설치 시 **"Add Python to PATH"** 체크 필수!
3. 설치 확인:
    ```bash
    python --version
    # Python 3.11.x 출력되면 성공
    ```

#### 2.2 NVIDIA GPU 드라이버 (권장)

딥러닝 추론을 위해 NVIDIA GPU가 있다면:

1. [NVIDIA 드라이버](https://www.nvidia.com/download/index.aspx)에서 최신 드라이버 설치
2. 확인:
    ```bash
    nvidia-smi
    ```

#### 2.3 CUDA Toolkit (GPU 사용 시)

PyTorch가 GPU를 사용하려면 CUDA가 필요합니다.

-   [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) 권장

---

## 3. 환경 설정

### 3.1 가상환경 생성 (권장)

가상환경을 사용하면 프로젝트별로 패키지를 분리 관리할 수 있습니다.

```bash
# 프로젝트 폴더로 이동
cd "폴더경로\SSISO_API"

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
.\venv\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
# source venv/bin/activate
```

활성화되면 터미널에 `(venv)`가 표시됩니다.

### 3.2 필수 패키지 설치

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 필수 패키지 설치 (시간이 오래 걸릴 수 있음)
pip install -r requirements.txt
```

**주의**: PyTorch는 requirements.txt에 포함되어 있지만, GPU 버전이 필요하면 별도 설치:
https://pytorch.org/get-started/locally/ 에서 CUDA 버전에 맞는 torch 선택 가능

```bash
# GPU 버전 PyTorch 설치 (CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3.3 nnUNet 설치 (선택사항)

nnUNet 모델을 사용하려면 추가 설치가 필요합니다:

```bash
pip install nnunetv2
```

---

## 4. 프로젝트 구조

```
SSISO_API/
├── main.py                 # 메인 API 서버 파일
├── requirements.txt        # 필요한 패키지 목록
├── utils/
│   └── util.py            # 유틸리티 함수 (파일 변환 등)
├── runs/
│   ├── run_model.py       # 모델 실행 함수들
│   ├── [SEG] HCC-PV.py    # 개별 모델 학습 스크립트
│   ├── [SEG] HCC-PV-Infer.py  # 개별 모델 추론 스크립트
│   └── models/            # 학습된 모델 가중치 (.pth 파일)
├── uploads/               # 업로드된 파일 저장 (자동 생성)
├── converted/             # 변환된 파일 저장 (자동 생성)
├── results/               # 결과 파일 저장 (자동 생성)
└── temp/                  # 임시 파일 저장 (자동 생성)
```

---

## 5. 서버 실행

### 5.1 기본 실행

```bash
# 가상환경이 활성화된 상태에서
uvicorn main:app --reload
```

성공하면 다음과 같은 메시지가 출력됩니다:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### 5.2 다른 포트로 실행

```bash
uvicorn main:app --reload --port 8080
```

### 5.3 외부 접속 허용

다른 컴퓨터에서 접속하려면:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 6. API 사용 방법

### 6.1 웹 브라우저로 접속

서버 실행 후 브라우저에서:

-   **메인 페이지**: http://localhost:8000
-   **API 문서 (Swagger)**: http://localhost:8000/docs
-   **상태 확인**: http://localhost:8000/health

### 6.2 API 엔드포인트

| 엔드포인트                        | 설명                   | 입력         | 출력              |
| --------------------------------- | ---------------------- | ------------ | ----------------- |
| `POST /nifti-to-nrrd/`            | NIfTI → NRRD 변환      | .nii.gz 파일 | .nrrd 파일        |
| `POST /nrrd-to-nifti/`            | NRRD → NIfTI 변환      | .nrrd 파일   | .nii.gz 파일      |
| `POST /infer/hcc-pv/`             | HCC 문맥 영역 분석     | MRI .nii.gz  | 세그멘테이션 결과 |
| `POST /infer/ldlt-pvhv/`          | LDLT 문맥/간정맥 분석  | CT .nii.gz   | 세그멘테이션 결과 |
| `POST /infer/liver/`              | 간/혈관/비장 분석      | CT .nii.gz   | 세그멘테이션 결과 |
| `POST /infer/pancreas/`           | 췌장암 분석            | CT .nii.gz   | 세그멘테이션 결과 |
| `POST /infer/parenchyma/`         | 실질/암 분석           | MRI .nii.gz  | 세그멘테이션 결과 |
| `POST /infer/liver_5sect_nnunet/` | 간 5구역 분석 (nnUNet) | CT .nii.gz   | 세그멘테이션 결과 |
| `POST /infer/kidney_nnunet/`      | 신장 분석 (nnUNet)     | CT .nii.gz   | 세그멘테이션 결과 |

### 6.3 curl 사용 예시

```bash
# 파일 변환 (NIfTI → NRRD)
curl -X POST -F "file=@input.nii.gz" http://localhost:8000/nifti-to-nrrd/ --output output.nrrd

# HCC-PV 모델로 추론 (NRRD 출력)
curl -X POST -F "file=@mri_image.nii.gz" "http://localhost:8000/infer/hcc-pv/?output_format=.nrrd" --output result.nrrd

# HCC-PV 모델로 추론 (NIfTI 출력)
curl -X POST -F "file=@mri_image.nii.gz" "http://localhost:8000/infer/hcc-pv/?output_format=.nii.gz" --output result.nii.gz
```

### 6.4 Python으로 API 호출

```python
import requests

# 파일 변환 예시
url = "http://localhost:8000/nifti-to-nrrd/"
files = {"file": open("input.nii.gz", "rb")}
response = requests.post(url, files=files)

with open("output.nrrd", "wb") as f:
    f.write(response.content)

# 모델 추론 예시
url = "http://localhost:8000/infer/hcc-pv/"
files = {"file": open("mri_image.nii.gz", "rb")}
params = {"output_format": ".nrrd"}
response = requests.post(url, files=files, params=params)

with open("result.nrrd", "wb") as f:
    f.write(response.content)
```

---

## 7. 문제 해결

### 7.1 "ModuleNotFoundError" 오류

패키지가 설치되지 않은 경우:

```bash
pip install <패키지명>
```

### 7.2 CUDA 관련 오류

GPU를 인식하지 못하는 경우:

```python
import torch
print(torch.cuda.is_available())  # True여야 함
print(torch.cuda.get_device_name(0))  # GPU 이름 출력
```

False가 출력되면:

1. NVIDIA 드라이버 재설치
2. CUDA Toolkit 재설치
3. GPU 버전 PyTorch 재설치

### 7.3 "KMP_DUPLICATE_LIB_OK" 경고

이미 main.py에서 처리됨:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### 7.4 포트 충돌

8000번 포트가 이미 사용 중인 경우:

```bash
uvicorn main:app --port 8001
```

### 7.5 메모리 부족 (Out of Memory)

GPU 메모리가 부족한 경우:

-   더 작은 영상 파일 사용
-   다른 GPU 프로세스 종료
-   CPU 모드로 실행 (느림)

---

## 8. 추가 정보

### 8.1 모델 파일 위치

학습된 모델 가중치 파일(.pth)은 다음 경로에 있어야 합니다:

```
runs/models/<모델명>/<가중치파일>.pth
```

예시:

```
runs/models/HCC-PV-MRI-PP/559_0.3119_0.7134.pth
```

### 8.2 API 문서

더 자세한 API 문서는 서버 실행 후 다음 주소에서 확인:

-   **Swagger UI**: http://localhost:8000/docs
-   **ReDoc**: http://localhost:8000/redoc

### 8.3 개발자 연락처

-   **개발**: SSISO Lab
-   **버전**: 2.0.0

---

## 9. 빠른 시작 요약

```bash
# 1. 프로젝트 폴더로 이동
cd "c:\Users\Ryzen 5800X\Downloads\SSISO_API"

# 2. 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 서버 실행
uvicorn main:app --reload

# 5. 브라우저에서 http://localhost:8000 접속
```
