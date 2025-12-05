# dicom-server

Flask 기반의 의료 영상 파일(DICOM, NIfTI, NRRD 등)을 업로드하고, 웹 뷰어에서 시각화할 수 있도록 정적 서빙 및 처리를 지원하는 서버입니다.

## 📁 프로젝트 구조

dicom-server/
├── app.py                  # Flask 서버 진입점
├── requirements.txt        # Python 패키지 의존성 목록
├── uploads/                # 업로드된 의료 영상 저장 폴더
├── venv/                   # 가상환경 디렉토리 (.gitignore에 추가 필요)

## ✅ 사전 조건

- Python 3.9 이상
- macOS, Ubuntu, WSL2 환경에서 테스트됨

## ⚙️ 설치 및 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/your-username/dicom-server.git
cd dicom-server

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python app.py

## ./run.sh 실행
./run.sh

# Docker registry
495599733399.dkr.ecr.ap-northeast-2.amazonaws.com/isw-server

## Role 생성
```bash
# 예시: App Runner용 ECR 읽기 롤 생성
aws iam create-role --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow","Principal": {"Service": "build.apprunner.amazonaws.com"},"Action": "sts:AssumeRole"}]
  }'

aws iam attach-role-policy --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

## App Runner
```bash
export AWS_REGION=ap-northeast-2
export ACCOUNT_ID=495599733399
export IMAGE_URI=495599733399.dkr.ecr.ap-northeast-2.amazonaws.com/isw-server:latest
export SERVICE_NAME=isw-server
export SERVICE_ROLE_ARN=arn:aws:iam::495599733399:role/AppRunnerECRAccessRole
export ACCESS_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/AppRunnerECRAccessRole

aws apprunner create-service \
  --region "$AWS_REGION" \
  --service-name "$SERVICE_NAME" \
  --source-configuration 'ImageRepository={ImageIdentifier='"$IMAGE_URI"',ImageRepositoryType=ECR,ImageConfiguration={Port=5051}},AuthenticationConfiguration={AccessRoleArn='"$ACCESS_ROLE_ARN"'}' \
  --instance-configuration "Cpu=1 vCPU,Memory=2 GB" \
  --profile zerosketch
  ```