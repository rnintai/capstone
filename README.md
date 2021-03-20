# capstone

## 특정 경로의 파일 가져오기
import glob

import os.path



myPath = '/내가/원하는/디렉토리/경로'

myExt = '*.jpg' # 찾고 싶은 확장자



for a in glob.glob(os.path.join(myPath, myExt)):

    print(a)



출처: [크롬망간이 글 쓰는 공간](https://crmn.tistory.com/47)
