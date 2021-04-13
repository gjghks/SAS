# **S**ewer **AI** **S**olution

AI 기반 하수관로 관리 솔루션

#### SAS (v0.1) released ([Chanelog] (CHANGELOG.md))
 - Instance segmenation 방법 기반의 하수관로 결함 영역 자동 검출
 - 환경부 매뉴얼 기준 24개 + 1개(etc)로 총 25개의 결함 항목 분류
 - 촬영거리 기준 12834.79m에 해당하는 과거 하수관로 조사자료를 학습 데이터로 가공 및 학습, 지속적으로 추가 예정
 - bbox mAP(.5~.95): 39.11, mask mAP(.5~.95): 31.85
 - 30 frame 단위로 결함 정보 자동 tagging(원본/결과 이미지 및 결함 정보 csv 파일 생성)
 - 촬영거리 ocr 적용(고정 ROI)
 - 5600X, 32GB, RTX3070 8GB 기준, 빠른 추론 시간(약 20fps 이상)
 - Flask를 활용한 Local Server 기반의 web application
 - Image 및 Video 입력
 - Image 및 Video 출력

SAS의 예제 이미지들:

![Example 0](data/SAS_example_0.gif)


# To do List
 - [ ] 결함 조사 보고서 자동 생성 모듈 개발 
 - [ ] 촬영거리 ocr을 위한 text detection
 - [ ] 촬영거리의 다양한 폰트에 대한 tesseract 학습
 - [ ] 결함 정보 기반의 통계 및 분석 결과 도출
 - [ ] Flask-SQLAlchemy를 활용한 DB 추가
 - [ ] Heroku를 활용한 cloud platform 적용 for Demo
  
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