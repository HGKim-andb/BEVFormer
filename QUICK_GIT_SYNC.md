# Quick Git Sync for Full Dataset

Git을 사용한 간단한 동기화 방법입니다.

## 1. 소스 서버 (sp@katech)에서 실행

```bash
cd ~/Project/BEVFormer

# 파일 크기 확인 (중요!)
ls -lh data/emergence_risk_v5_full/*.pkl
du -sh data/emergence_risk_v5_full/

# Git 상태 확인
git status
```

### 파일 크기에 따른 처리:

#### Case A: 파일이 100MB 미만인 경우 (Git에 직접 추가 가능)

```bash
# Git에 추가
git add data/emergence_risk_v5_full/*.pkl
git add data/emergence_risk_v5_full/*.json

# 커밋
git commit -m "Add full dataset risk labels

- risk_labels_train.pkl: full dataset training labels
- risk_labels_val.pkl: full dataset validation labels
- risk_config.json: configuration"

# 푸시
git push origin master
```

#### Case B: 파일이 100MB 이상인 경우 (Git LFS 필요)

```bash
# Git LFS 설치 확인
git lfs version

# Git LFS 초기화 (처음 한번만)
git lfs install

# pkl 파일을 LFS로 추적
git lfs track "*.pkl"
git add .gitattributes

# 파일 추가
git add data/emergence_risk_v5_full/*.pkl
git add data/emergence_risk_v5_full/*.json

# 커밋 & 푸시
git commit -m "Add full dataset risk labels with Git LFS"
git push origin master
```

#### Case C: 파일이 너무 큰 경우 (2GB+)

pkl 파일을 Git에서 제외하고, 코드와 설정만 동기화:

```bash
# .gitignore에 추가
echo "data/emergence_risk_v5_full/*.pkl" >> .gitignore

# 코드와 설정 파일만 커밋
git add projects/configs/bevformer/bevformer_risk_tiny.py
git add tools/*.py
git add *.md
git commit -m "Update configs for full dataset"
git push origin master
```

이 경우 pkl 파일은 별도로 전송 (rsync 또는 scp 사용)

---

## 2. 타겟 서버 (hg-main@hgmain-Z490)에서 실행

```bash
cd /home/hg-main/data2/BEVFormer

# 현재 변경사항 확인
git status

# 변경사항이 있으면 커밋 또는 stash
git add -A
git commit -m "Update local changes"
# 또는
git stash

# Pull
git pull origin master

# Git LFS 사용한 경우
git lfs pull
```

---

## 3. 확인

```bash
# 파일 확인
ls -lh data/emergence_risk_v5_full/

# 통계 확인
PYTHONPATH=.:$PYTHONPATH /home/hg-main/anaconda3/envs/vad1/bin/python tools/check_risk_labels.py
```

---

## 권장사항

### 파일 크기별 권장 방법:

| 파일 크기 | 방법 | 명령어 |
|---------|------|--------|
| < 50MB | 일반 Git | `git add` + `git push` |
| 50MB - 2GB | Git LFS | `git lfs track "*.pkl"` |
| > 2GB | 별도 전송 | `rsync` 또는 공유 스토리지 |

### Git LFS 설치 (필요시)

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# CentOS/RHEL
sudo yum install git-lfs

# macOS
brew install git-lfs
```

---

## 빠른 동기화 스크립트

소스 서버에서:

```bash
#!/bin/bash
# sync_to_git.sh

cd ~/Project/BEVFormer

# 파일 크기 확인
SIZE=$(du -m data/emergence_risk_v5_full/risk_labels_train.pkl | cut -f1)

if [ $SIZE -lt 100 ]; then
    echo "파일 크기: ${SIZE}MB - 일반 git 사용"
    git add data/emergence_risk_v5_full/*.pkl
    git add data/emergence_risk_v5_full/*.json
else
    echo "파일 크기: ${SIZE}MB - Git LFS 사용"
    git lfs track "*.pkl"
    git add .gitattributes
    git add data/emergence_risk_v5_full/*.pkl
    git add data/emergence_risk_v5_full/*.json
fi

# 코드 업데이트도 추가
git add projects/configs/bevformer/bevformer_risk_tiny.py
git add tools/*.py
git add *.md

git commit -m "Add full dataset risk labels and updated configs"
git push origin master

echo "동기화 완료!"
```

타겟 서버에서:

```bash
#!/bin/bash
# pull_from_git.sh

cd /home/hg-main/data2/BEVFormer

# 로컬 변경사항 저장
git stash

# Pull
git pull origin master
git lfs pull 2>/dev/null || true

# 확인
ls -lh data/emergence_risk_v5_full/
PYTHONPATH=.:$PYTHONPATH /home/hg-main/anaconda3/envs/vad1/bin/python tools/check_risk_labels.py

echo "동기화 완료!"
```

---

## Troubleshooting

### "file is 100MB; this exceeds GitHub's file size limit"

```bash
# Git LFS 사용
git lfs track "*.pkl"
git add .gitattributes
git add data/emergence_risk_v5_full/*.pkl
git commit --amend
git push origin master --force
```

### "This repository is over its data quota"

GitHub LFS는 1GB/month 무료. 초과시:
1. GitLab 사용 (10GB 무료)
2. 자체 Git 서버 사용
3. pkl 파일 제외하고 별도 전송

### ".pkl file corrupted after pull"

```bash
# Git LFS 재다운로드
git lfs fetch --all
git lfs checkout
```
