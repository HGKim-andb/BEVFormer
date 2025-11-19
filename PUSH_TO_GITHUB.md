SH가 안되면 Personal Access Token(PAT)을 사용하겠습니다.
Update Todos

Setup GitHub authentication with PAT
Personal Access Token으로 Push하기
1단계: GitHub에서 토큰 생성
GitHub 웹사이트 접속
Settings (우측 상단 프로필 → Settings)
Developer settings (좌측 맨 아래)
Personal access tokens → Tokens (classic)
Generate new token → Generate new token (classic)
설정:
Note: BEVFormer-Risk-Push
Expiration: 90 days
Scopes: ✅ repo (전체 체크)
Generate token 클릭
토큰 복사 (한번만 보여짐!)
2단계: Git Credential 설정
토큰을 생성하셨으면, 아래 명령어를 실행하세요:
# Credential helper 설정
git config --global credential.helper store
3단계: Push (토큰 입력)
git push my-repo master
그러면 Username과 Password를 물어봅니다:
Username for 'https://github.com': HGKim-andb
Password for 'https://HGKim-andb@github.com': [생성한 토큰 붙여넣기]
📝 토큰 생성 전이면:
토큰이 아직 없으시면 먼저 생성하세요. 제가 credential helper만 먼저 설정해드릴게요:
Bash
⎿
Enable credential storage
IN
git config --global credential.helper store
OUT
0
설정 완료! 이제 다음 단계:
GitHub에서 Personal Access Token 생성 (위 1단계 참고)
토큰을 복사해두세요
그 다음 알려주시면 push 명령어를 실행하겠습니다
토큰 생성하셨나요?