import re

# nohup.out 파일 경로
file_path = "nohup.out"

# 저장할 결과 파일 경로
output_file_path = "output.txt"

# 두 가지 JSON 형식을 추출하기 위한 정규표현식 패턴 정의
pattern = r"\{'(?:eval_loss|loss)': [^}]+, 'epoch': [^\}]+\}"

# 결과를 저장할 리스트
results = []

# 파일 읽기 및 추출
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        matches = re.findall(pattern, line)
        results.extend(matches)

# 추출된 결과를 하나의 문자열로 결합
output_text = '\n'.join(results)

# 추출된 결과를 파일로 저장
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(output_text)

print(f"추출된 텍스트가 '{output_file_path}' 파일에 저장되었습니다.")
