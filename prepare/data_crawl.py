import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# 지역 코드 매핑 (소문자로 표기하여 대소문자 구분 없이 매칭)
region_codes = {
    "seoul": "11",
    "busan": "21",
    "daegu": "22",
    "incheon": "23",
    "gwangju": "24",
    "daejeon": "25",
    "ulsan": "26",
    "gyeonggi-do": "31",
    "gangwon-do": "32",
    "chungcheongbuk-do": "33",
    "chungcheongnam-do": "34",
    "jeollabuk-do": "35",
    "jeollanam-do": "36",
    "gyeongsangbuk-do": "37",
    "gyeongsangnam-do": "38"
}

# 문화재 종류 코드 매핑
classification_codes = {
    "National Treasure": "11",
    "Treasure": "12",
    "Historic Site": "13"
}

# ccebAsno 규칙을 적용하여 계산하는 함수
def calculate_ccebAsno(no):
    if 1 <= no <= 47:
        return f"{no:04}0000"
    elif no == 48 :
        return "00480100"
    elif no == 49 :
        return "00480200"   
    elif 50 <= no <= 137:
        return f"{(no - 1):04}0000"
    elif no == 138 :
        return "01370100"
    elif no == 139 :
        return "01370200"
    elif 140 <= no <= 149:
        return f"{(no - 2):04}0000"
    elif no == 150 :
        return "01480100"
    elif no == 151 :
        return "01480200"
    elif no == 152 :
        return "01490100"
    elif no == 153 :
        return "01490200"
    elif no == 154 :
        return "01500000"
    elif no == 155:
        return "01510100"
    elif no == 156:
        return "01510200"
    elif no == 157:
        return "01510300"
    elif no == 158:
        return "01510400"
    elif no == 159:
        return "01510500"
    elif no == 160:
        return "01510600"
    elif 161 <= no <= 176:
        return f"{(no - 9):04}0000"
    elif 177 <= no <= 240:
        return f"{(no - 8):04}0000"
    elif no == 241:
        return "02330100"
    elif no == 242:
        return "02330200"
    elif 243 <= no <= 257:
        return f"{(no - 9):04}0000"
    elif no == 258:
        return "02490100"
    elif no == 259:
        return "02490200"
    elif 260 <= no <= 283:
        return f"{(no - 10):04}0000"
    elif 284 <= no <= 286:
        return f"{(no - 9):04}0000"
    elif 287 <= no <= 313:
        return f"{(no - 8):04}0000"
    elif no == 314:
        return "03060000"
    elif no == 315:
        return "03060200"
    elif no == 316:
        return "03060300"
    elif no == 317:
        return "03060400"
    elif 318 <= no <= 329:
        return f"{(no - 11):04}0000"
    elif no == 330:
        return "03190100"
    elif no == 331:
        return "03190200"
    elif no == 332:
        return "03190300"
    elif 333 <= no <= 334:
        return f"{(no - 13):04}0000"
    elif no == 335:
        return "03220100"
    elif no == 336:
        return "03220200"
    elif 337 <= no <= 356:
        return f"{(no - 14):04}0000"

    else:
        return None

# CSV 파일에서 데이터 읽기
data = pd.read_csv("./newdataset/korea_heritage_service/performance_reservation_list.csv")

# 상위 356개 항목 선택
test_data = data.head(356).to_dict(orient="records")

# 결과를 저장할 리스트
processed_data = []

# 성공적으로 저장된 항목 수
success_count = 0

def get_region_code(address):
    # Address에서 첫 번째 단어만 추출해 소문자로 변환 후 지역 코드 매핑
    region = address.split(" ")[0].lower()
    return region_codes.get(region, "Unknown") 

def save_description_and_image(no, name, folder_path, ccebKdcd, ccebCtcd, ccebAsno):
    search_url = (
        f"https://english.khs.go.kr/chaen/search/selectGeneralSearchDetail.do?"
        f"mn=EN_02_02&sCcebKdcd={ccebKdcd}&ccebAsno={ccebAsno}&sCcebCtcd={ccebCtcd}"
        f"&pageIndex=1&region=&canAsset=&ccebPcd1=&searchWrd={name.replace(' ', '+')}"
    )
    print(f"Processing No: {no}, Name: {name}, Search URL: {search_url}")

    response = requests.get(search_url)
    
    description_path = os.path.join(folder_path, f"{no}_description.txt")
    description_saved = False
    img_url = None
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 설명 가져오기
        content_div = soup.find("div", {"class": "hschDetail_con"})
        if content_div:
            description = content_div.find("p").get_text(strip=True)
            with open(description_path, 'w', encoding='utf-8') as file:
                file.write(description)
            description_saved = True
            print(f"{no} description saved.")
        else:
            print(f"{no} description unsaved.")
        
        # 이미지 URL 가져오기
        img_div = soup.find("div", {"class": "hschDi_img"})
        if img_div:
            img_tag = img_div.find("img")
            if img_tag and 'src' in img_tag.attrs:
                img_url = img_tag['src']
                if img_url.startswith("//"):
                    img_url = "https:" + img_url

                save_image(no, img_url, folder_path)
            else:
                print(f"{no} image not found in HTML.")
        else:
            print(f"{no} image div not found.")
    else:
        print(f"Can't enter to {no} data page error code: {response.status_code}")

    return description_saved, img_url, search_url

def save_image(no, img_url, folder_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    img_path = os.path.join(folder_path, f"{no}.jpg")
    try:
        response = requests.get(img_url, headers=headers)
        if response.status_code == 200:
            with open(img_path, 'wb') as file:
                file.write(response.content)
            print(f"{no} image saved.")
            return img_path
        else:
            print(f"{no} image unsaved : {response.status_code}")
    except Exception as e:
        print(f"{no} image error code: {e}")
    return None

def process_data(data):
    global success_count
    
    for item in data:
        no = item["No"]
        name = item["Name of Cultural Heritage"].split(",")[0] 
        classification = item["Classification"]
        address = item["Address"]
        
        # 광역 지역 코드 및 문화재 종류 코드 설정
        ccebCtcd = get_region_code(address)
        ccebKdcd = classification_codes.get(classification, "Unknown")
        ccebAsno = calculate_ccebAsno(no)

        # 각 데이터 폴더 생성 (이미 존재하면 건너뜀)
        folder_path = os.path.join("newdataset", "newdata", str(no))
        if os.path.exists(folder_path):  
            print(f"{no} already exists. Skipping download.")
            continue
        
        os.makedirs(folder_path, exist_ok=True)
        
        # 설명 및 이미지 저장
        description_saved, img_url, search_url = save_description_and_image(no, name, folder_path, ccebKdcd, ccebCtcd, ccebAsno)
        
        # 성공적으로 저장된 경우 카운트 증가
        if description_saved and img_url:
            success_count += 1
        
        # CSV에 저장할 항목 구성
        processed_data.append({
            "No": no,
            "Classification": classification,
            "Name of Cultural Heritage": item["Name of Cultural Heritage"],
            "Korean": item["Korean"],
            "ccebAsno": ccebAsno,
            "heritage_url": search_url,
            "img": img_url  
        })

# 폴더 내의 데이터를 바탕으로 CSV 파일을 재작성하는 함수
def reconstruct_csv(data):
    processed_data = []
    
    for item in data:
        no = item["No"]
        classification = item["Classification"]
        name = item["Name of Cultural Heritage"]
        korean_name = item["Korean"]
        
        # 각 데이터 폴더의 경로
        folder_path = os.path.join("newdataset", "newdata", str(no))
        
        # 설명 파일 확인
        description_path = os.path.join(folder_path, f"{no}_description.txt")
        description_saved = os.path.exists(description_path)
        
        # 이미지 파일 확인
        img_path = os.path.join(folder_path, f"{no}.jpg")
        img_url = img_path if os.path.exists(img_path) else None
        
        # URL 구성
        ccebAsno = calculate_ccebAsno(no)
        heritage_url = (
            f"https://english.khs.go.kr/chaen/search/selectGeneralSearchDetail.do?"
            f"mn=EN_02_02&sCcebKdcd={classification_codes.get(classification, 'Unknown')}&"
            f"ccebAsno={ccebAsno}&sCcebCtcd={get_region_code(item['Address'])}&"
            f"pageIndex=1&region=&canAsset=&ccebPcd1=&searchWrd={name.replace(' ', '+')}"
        )
        
        # CSV에 저장할 항목 구성
        processed_data.append({
            "No": no,
            "Classification": classification,
            "Name of Cultural Heritage": name,
            "Korean": korean_name,
            "ccebAsno": ccebAsno,
            "heritage_url": heritage_url,
            "img": img_url  
        })
    
    # CSV 파일로 저장
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv("processing_data.csv", index=False, encoding="utf-8-sig")
    print(f"CSV 파일이 'processing_data.csv'로 재작성되었습니다.")

# 상위 356개 항목 선택
test_data = data.head(356).to_dict(orient="records")

# CSV 파일만 재작성
reconstruct_csv(test_data)

# # 테스트 실행
# process_data(test_data)

# # 처리한 데이터를 새로운 CSV 파일에 저장
# processed_df = pd.DataFrame(processed_data)
# processed_df.to_csv("processing_data.csv", index=False, encoding="utf-8-sig")

print(f"CSV 파일이 'processing_data.csv'로 저장되었습니다.")
print(f"성공적으로 저장된 항목 수: {success_count}")
