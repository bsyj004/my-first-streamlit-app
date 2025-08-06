import streamlit as st
from PIL import Image
import numpy as np
import mediapipe as mp

# 스타일 (따뜻한 베이지 + 미니멀)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
    .stApp {
        background: linear-gradient(135deg, #fdf6e3, #f0ead8);
        min-height: 100vh;
        box-shadow: inset 0 0 60px rgba(0,0,0,0.05);
    }
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #4a3c31;
        background: transparent;
    }
    .css-1d391kg {
        max-width: 700px;
        margin: auto;
        padding: 2.5rem 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transition: box-shadow 0.3s ease;
    }
    .css-1d391kg:hover {
        box-shadow: 0 12px 32px rgba(0,0,0,0.12);
    }
    h1, h2, h3, h4 {
        color: #3e2f1c;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .stButton>button {
        background-color: #b37f4e;
        color: white;
        border-radius: 10px;
        padding: 12px 26px;
        font-weight: 700;
        font-size: 1rem;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 3px 6px rgba(179,127,78,0.4);
    }
    .stButton>button:hover {
        background-color: #8e6233;
        box-shadow: 0 6px 12px rgba(142,98,51,0.5);
        cursor: pointer;
    }
    .result-box {
        border-radius: 14px;
        padding: 1.3rem 2rem;
        margin: 1.2rem 0;
        font-size: 1.15rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.07);
        letter-spacing: 0.02em;
    }
    .success-box {
        background-color: #e4f0d6;
        color: #4a682e;
        border-left: 6px solid #7bb342;
    }
    .info-box {
        background-color: #f9f1d1;
        color: #7d6b2c;
        border-left: 6px solid #d4a017;
    }
    .warning-box {
        background-color: #fdebd3;
        color: #a2691f;
        border-left: 6px solid #d97e00;
    }
    .error-box {
        background-color: #f9d1cc;
        color: #a23a3a;
        border-left: 6px solid #c53030;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "face_scores" not in st.session_state:
    st.session_state.face_scores = None
if "survey_data" not in st.session_state:
    st.session_state.survey_data = None
if "final_score" not in st.session_state:
    st.session_state.final_score = None

# 사이드바 메뉴
menu = st.sidebar.selectbox("메뉴 선택", ["Home", "얼굴 점수 계산", "설문조사", "결과"])

def face_score_page():
    st.header("얼굴 점수 계산기")
    uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(img_np)

            if not results.multi_face_landmarks:
                st.error("얼굴을 인식할 수 없습니다.")
                st.session_state.face_scores = None
                return

            face_landmarks = results.multi_face_landmarks[0]

            h, w, _ = img_np.shape
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

            # 왼쪽 눈 양 끝 점 (33, 133), 오른쪽 눈 양 끝 점 (362, 263)
            left_eye_width = np.linalg.norm(landmarks[33] - landmarks[133])
            right_eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
            eye_size = (left_eye_width + right_eye_width) / 2

            # 코 높이: 코 중앙 윗부분(6)과 코끝(2) 거리
            nose_height = np.linalg.norm(landmarks[6] - landmarks[2])

            # mediapipe 좌표 기준 max 값 (적절히 조절 가능)
            MAX_EYE_SIZE = 80
            MAX_NOSE_HEIGHT = 60

            eye_score = min(eye_size / MAX_EYE_SIZE, 1.0) * 100
            nose_score = min(nose_height / MAX_NOSE_HEIGHT, 1.0) * 100
            total_face_score = round((eye_score + nose_score) / 2, 1)

            st.image(img, caption="업로드한 이미지", use_column_width=True)
            st.write(f"눈 점수: {eye_score:.1f} %")
            st.write(f"코 점수: {nose_score:.1f} %")
            st.write(f"얼굴 점수 총합: {total_face_score} %")

            st.session_state.face_scores = {
                "eye_score": eye_score,
                "nose_score": nose_score,
                "total_face_score": total_face_score
            }
    else:
        st.info("사진을 업로드하면 얼굴 점수를 계산합니다.")

def survey_page():
    st.header("설문조사")

    if st.session_state.face_scores is None:
        st.warning("📸 얼굴 점수를 먼저 계산해 주세요!")
        st.stop()

    choice = st.radio("성적 만족도 또는 연봉 중 하나를 선택하세요:", ["성적 만족도 (1~10)", "연봉 (만원 단위)"])

    with st.form("survey_form"):
        height = st.number_input("키(cm)", min_value=100, max_value=250, step=1)
        weight = st.number_input("몸무게(kg)", min_value=30, max_value=200, step=1)
        personality = st.selectbox("성격", ["외향적", "내향적", "중립적"])

        sexuality = None
        salary = None

        if choice == "성적 만족도 (1~10)":
            sexuality = st.slider("성적 만족도", 1, 10, 5)
        else:
            salary = st.number_input("연봉 (만원)", min_value=1000, max_value=10000, step=100)

        submitted = st.form_submit_button("설문 제출")

    if submitted:
        st.success("설문 제출 완료!")
        st.session_state.survey_data = {
            "height": height,
            "weight": weight,
            "personality": personality,
            "sexuality": sexuality,
            "salary": salary
        }

def result_page():
    st.header("결과")

    face_scores = st.session_state.face_scores
    survey_data = st.session_state.survey_data

    if face_scores is None:
        st.warning("먼저 '얼굴 점수 계산'에서 사진을 업로드하고 점수를 계산하세요.")
        return
    if survey_data is None:
        st.warning("먼저 '설문조사'에서 설문을 완료하세요.")
        return

    face_total_50 = face_scores["total_face_score"] * 0.5
    height_score = (survey_data["height"] / 250) * 25  # 키는 25점 만점

    if survey_data["sexuality"] is not None:
        sexuality_score = (survey_data["sexuality"] / 10) * 25  # 성적 만족도 25점 만점
        survey_score_50 = height_score + sexuality_score
    else:
        salary = survey_data["salary"]
        salary_score = ((salary - 1000) / (10000 - 1000)) * 25  # 연봉 25점 만점
        survey_score_50 = height_score + salary_score

    final_score = face_total_50 + survey_score_50
    st.session_state.final_score = final_score

    st.subheader("점수 요약")
    results = [
        f"눈 점수: {face_scores['eye_score']:.1f} %",
        f"코 점수: {face_scores['nose_score']:.1f} %",
        f"얼굴 점수 (50점 만점): {face_total_50:.1f} 점",
        f"키: {survey_data['height']} cm",
        f"몸무게: {survey_data['weight']} kg",
        f"성격: {survey_data['personality']}",
    ]

    if survey_data["sexuality"] is not None:
        results.append(f"성적 만족도: {survey_data['sexuality']}")
        results.append(f"설문 점수 (50점 만점): {survey_score_50:.1f} 점")
    else:
        results.append(f"연봉: {survey_data['salary']} 만원")
        results.append(f"설문 점수 (50점 만점): {survey_score_50:.1f} 점")

    results.append(f"여친이 생길 가능성 (100점 만점): {final_score:.1f} 점")

    for item in results:
        st.markdown(f"- {item}")

    if final_score >= 80:
        st.markdown('<div class="result-box success-box">여친이 생길 가능성 매우 높음!</div>', unsafe_allow_html=True)
    elif final_score >= 60:
        st.markdown('<div class="result-box info-box">꽤 좋은 편이에요. 조금만 더 노력해봐요!</div>', unsafe_allow_html=True)
    elif final_score >= 40:
        st.markdown('<div class="result-box warning-box">보통이에요. 매력을 더 키워보세요!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box error-box">노력과 자기계발이 필요합니다. 화이팅!</div>', unsafe_allow_html=True)

def home_page():
    st.header("💕 여친 생길 가능성 분석기 💕")
    st.markdown("""
    안녕하세요!  
    이 앱은 당신의 얼굴 사진과 간단한 설문조사를 바탕으로  
    여친이 생길 가능성을 점수로 알려드립니다.

    **사용법**  
    1. '얼굴 점수 계산'에서 사진 업로드 후 얼굴 점수를 계산하세요.  
    2. '설문조사'에서 본인의 정보를 입력하세요.  
    3. '결과'에서 종합 점수와 해석을 확인하세요.

    **Tip**  
    - 설문조사에서 ‘성적 만족도’ 혹은 ‘연봉’ 중 편한 항목을 선택할 수 있습니다.  
    - 점수는 0~100점 만점이며, 높을수록 여친이 생길 가능성이 큽니다.  

    즐거운 시간 되세요! 😊
    """)
    st.markdown("___")
    st.markdown("📸 ➡️ 📝 ➡️ 📊 순서로 진행해 주세요!")

# 메뉴에 따른 페이지 표시
if menu == "Home":
    home_page()
elif menu == "얼굴 점수 계산":
    face_score_page()
elif menu == "설문조사":
    survey_page()
elif menu == "결과":
    result_page()
