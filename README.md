# **S**ewer **AI** **S**olution

AI 기반 하수관로 관리 솔루션

#### SAS (v0.1) released ([Chanelog] (CHANGELOG.md))
- Instance segmenation 방법 기반의 하수관로 결함 영역 자동 검출
- 환경부 매뉴얼 기준 24개 + 1개(etc)로 총 25개의 결함 항목 분류
- 촬영거리 기준 12834.79m에 해당하는 과거 하수관로 조사자료를 학습 데이터로 가공 및 학습, 지속적으로 추가 예정
- 30 frame 단위로 결함 정보 자동 tagging(원본/결과 이미지 및 결함 정보 csv 파일 생성)
- 5600X, 32GB, RTX3070 8GB 기준, 빠른 분석 시간(약 20fps)
- Flask를 활용한 Local Server 기반의 web application
- Image 및 Video 입력
- Image 및 Video 출력


# Citation
## If you use SAS in your work, please citing below.
If you use YOLACT or this code base in your work, please cite.
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```


# contact
For questions about code, please contact [Hwan Heo](mailto:hheo@bizdata.kr).