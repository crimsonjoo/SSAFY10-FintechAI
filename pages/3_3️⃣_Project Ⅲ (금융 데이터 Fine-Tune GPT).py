import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np
from io import BytesIO
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components
import time
import graphviz
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import datetime
import replicate
from streamlit_drawable_canvas import st_canvas
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma





def init(): # Web App 설정
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI의 API 키를 설정해주세요.")
        exit(1)
    else:
        print("OPENAI의 API 키를 성공적으로 적용했습니다!")

    st.set_page_config(
        page_title="SAFFY 금융/경제 지식교육 GPT"
    )




@st.cache_resource(experimental_allow_widgets=True)
def bot_message(assistant_response):
    message_placeholder = st.empty()
    full_response = ""
    assistant_response = assistant_response
    # Simulate stream of response with milliseconds delay
    for chunk in assistant_response.split():
        full_response += chunk + " "
        time.sleep(0.15)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": full_response})



# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input,gpt_type,temperature,top_p,max_length):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"

    if gpt_type == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif gpt_type == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

    
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output





# def qa_gpt(replicate_api,gpt_type,temperature,top_p,max_length):
#     # User-provided prompt
#     if prompt := st.chat_input(disabled=not replicate_api):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.write(prompt)

#     # Generate a new response if last message is not from assistant
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = generate_llama2_response(prompt)
#                 placeholder = st.empty()
#                 full_response = ''
#                 for item in response:
#                     full_response += item
#                     placeholder.markdown(full_response)
#                 placeholder.markdown(full_response)
#         message = {"role": "assistant", "content": full_response}
#         st.session_state.messages.append(message)



def gpt(user_name,user_date,service_type,replicate_api,gpt_type,temperature,top_p,max_length):
    year = user_date.strftime('%Y')
    month = user_date.strftime('%M')
    day = user_date.strftime('%D')

  
    with st.chat_message("system"):
        bot_message(f'챗봇 상담을 시작합니다.')

    with st.chat_message("assistant"):
        bot_message(f'안녕하세요!')
        bot_message(f'저는 {service_type} 상담 AI 음성봇입니다.')
        bot_message(f'{user_name} 고객님 본인 되십니까?')


    time.sleep(0.5)
    with st.chat_message("user"): # 1. 본인 확인         
        chatbot1= st.selectbox("사용자 응답", ('《Ⅰ. 본인 확인 여부》','네','아니오'))
        
    if chatbot1 == '아니오': # 1. 본인 확인 - (아니오)
        with st.chat_message("assistant"):
            bot_message('죄송합니다.')
            bot_message(f'{user_name} 고객님 본인에게만 상담이 가능합니다.')
            bot_message('상담을 종료하겠습니다. 감사합니다.')
        with st.chat_message("system"):
            bot_message(f'챗봇 상담이 종료되었습니다.')

    elif chatbot1 == '네': # 1. 본인 확인 - (네)
        with st.chat_message("assistant"):
            bot_message(f'네! {user_name} 고객님, 반갑습니다 😀')
            bot_message(f'저희 {service_type} 상품을 가입해주셔서 진심으로 감사드립니다.')
            bot_message('가입하신 상품의 중요한 사항이 제대로 설명되었는지, 확인드리고자 연락드렸습니다')
            bot_message('상담 예상 소요 시간은 5분입니다. 시간 괜찮으신가요?')

        time.sleep(0.5)
        with st.chat_message("user"): # 2. 통화 가능 여부            
            chatbot2 = st.selectbox("사용자 응답", ('《Ⅱ. 통화 가능 여부》','네','아니오'))

        if chatbot2 == '아니오': # 2. 통화 가능 여부 - (아니오)
            with st.chat_message("assistant"):
                bot_message('네, 알겠습니다. 지금은 상담이 어려우시군요.')
                bot_message('가능한 일정을 선택해 주시면, 상담 일정을 예약해드리겠습니다.')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('')
                d = st.date_input("예약 날짜", datetime.date(2023, 2, 3))

            with col2:
                st.subheader('')
                t = st.time_input('예약 시간', datetime.time(10, 00))

            st.title('')
            with st.chat_message("assistant"):
                bot_message(f'[ 예약 날짜 : {d} / 예약 시간 : {t} ]')
                bot_message(f'선택하신 일정으로 상담 예약을 확정해드릴까요?')
                

            with st.chat_message('user'):
                chatbot2_ = st.selectbox("사용자 응답", ('《선택하기》','네',))

            if chatbot2_ == '네':
                with st.chat_message("assistant"):
                    bot_message(f'네 알겠습니다!')
                    bot_message(f'선택하신 상담 예약일에 다시 뵙겠습니다.')
                    bot_message(f'상담을 종료하겠습니다.')
                with st.chat_message("assistant"):
                    bot_message(f'챗봇 상담이 종료되었습니다.')

        elif chatbot2 == '네': # 2. 통화 가능 여부 - (네)
            with st.chat_message("assistant"):
                bot_message('고객님, 귀한 시간 내주셔서 감사합니다! 😄')
                bot_message('지금부터 진행하는 내용은 고객님의 권리 보호를 위해 기록되며,')
                bot_message('답변하신 내용은 향후 민원 발생시, 중요한 근거자료로 활용되오니,')
                bot_message('정확한 답변 부탁드립니다.')
                bot_message('')
                bot_message('')
                
                bot_message(f'계약자와 피보험자가 다른 계약의 경우, {user_name} 고객님의')
                bot_message(f'계약체결에 대한 동의가 반드시 필요합니다.')
                bot_message(f'자필서명이 정확히 이루어지지 않은 경우, 무효계약으로 간주되어')
                bot_message(f'고객님께서 직접적인 불이익 또는 손해를 입으실 수 있습니다.')
                bot_message('고객님께서 청약서에 직접 자필서명을 하셨는지요?')

            time.sleep(0.5)
            with st.chat_message("user"): # 3. 자필 서명 여부            
                chatbot3 = st.selectbox("사용자 응답", ('《Ⅲ. 자필 서명 여부》','네','아니오'))

            if chatbot3 == '아니오': # 3. 자필 서명 여부 - 아니오
                with st.chat_message("assistant"):     
                    bot_message(f'{user_name} 고객님 본인의 서명을 부탁드립니다.')
                    bot_message('서명을 완료하신 후, 하단의 [제출하기] 버튼을 눌러주세요')

                    canvas_result = st_canvas(
                                                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                                                stroke_width=3,
                                                stroke_color='black',
                                                background_color= 'white',
                                                background_image= None,
                                                update_streamlit= False,
                                                height=150,
                                                width=500,
                                                drawing_mode='freedraw',
                                                point_display_radius=0,
                                                key="canvas",
                                            )
                    
                    if st.button('서명 제출하기'):
                        st.header('')
                        chatbot3 = '네'

                    
            if chatbot3 == '네': # 3. 자필 서명 여부 - 네
                with st.chat_message("assistant"):
                    bot_message('네! 확인 감사합니다. 마지막으로 질문 한 가지 드리겠습니다.')
                    bot_message('직업, 건강상태 등 계약 전 알려야 할 의무 사항을 속이거나')
                    bot_message('제대로 알리지 않아 발생할 수 있는 모든 불이익은 고객님께 귀속됩니다.')
                    bot_message('')
                    bot_message('')
                    bot_message('고객님께서는 해당 내용을 모두 정확히 확인하시고 작성하셨습니까?')

                time.sleep(0.5)
                with st.chat_message("user"): # [4. 내용 인지 확인]            
                    chatbot4 = st.selectbox("사용자 응답", ('《Ⅳ. 내용 인지 확인》','네','아니오'))

                if chatbot4 == '아니오': # 4. 내용 인지 확인 - 아니오
                    with st.chat_message("assistant"):   
                        bot_message('고객님의 정보를 정확히 확인하신 후 상담을 다시 요청해주세요.')
                        bot_message('상담을 종료하겠습니다.')
                    with st.chat_message("system"):   
                        bot_message('상담이 종료되었습니다.')

                elif chatbot4 == '네': # 4. 내용 인지 확인 - 네
                    with st.chat_message("assistant"):   
                        bot_message('소중한 시간 내주셔서 감사드립니다. 향후 불편하시거나 궁금하신점 있으시면,')
                        bot_message('담당자나 콜센터로 언제든지 연락주시기 바랍니다 😊')
                        bot_message('기타 문의하실 추가 사항이 있으신가요?')

                    time.sleep(0.5)
                    with st.chat_message("user"): # [5. 추가 질문]            
                        chatbot5 = st.selectbox("사용자 응답", ('《Ⅴ. 추가 질문 여부》','아니오','네'))

                    if chatbot5 == '아니오':
                        with st.chat_message("assistant"):   
                            bot_message('상담을 종료하겠습니다. 좋은 하루 되세요.')
                            bot_message('감사합니다 🤗')
                        with st.chat_message("system"):   
                            bot_message('상담이 종료되었습니다.')

                    elif chatbot5 == '네':                       
                        if "messages" not in st.session_state:
                            st.session_state.messages = [{{"role": "assistant", "content": "무엇을 도와드릴까요?"}}] 

                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.write(message["content"])

                        if prompt := st.chat_input(disabled=not replicate_api):
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            with st.chat_message("user"):
                                st.write(prompt)

                        # Generate a new response if last message is not from assistant
                        # if st.session_state.messages[-1]["role"] != "assistant":
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = generate_llama2_response(prompt,gpt_type,temperature,top_p,max_length)
                                placeholder = st.empty()
                                full_response = ''
                                for item in response:
                                    full_response += item
                                    placeholder.markdown(full_response)
                                placeholder.markdown(full_response)
                        message = {"role": "assistant", "content": full_response}
                        st.session_state.messages.append(message)
                        # # while 문 적용하기 ? --> Llama2 or Fine-tune GPT
                        # prompt = st.chat_input("> 원하는 질문 입력")
                        # if prompt:
                        #     with st.chat_message("user"):   
                        #         bot_message(prompt)














def PJT3():
    init()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Project III")
    st.subheader(" : 금융 데이터 Fine-tune GPT")
    st.markdown('- 명세서 개발자 : 주정수 joojs2004@gmail.com')
    
    # with st.sidebar:
    #     user_input = st.text_input("당신의 질문 : ", key="user_input")

    
    with st.sidebar:
        st.header('사용자 지정 입력')
        st.text('')

        
        user_name = st.text_input("😀 사용자 이름")
        st.caption('')

        user_date = st.date_input("📅 상품 가입일")
        st.caption('')

        service_type = st.selectbox("🎯 금융 분야", ('은행','보험','카드','증권'))
        st.caption('')

        chatbot_type= st.selectbox("🎯  서비스 부문", ('완전판매 모니터링',))
        st.caption('')

        st.header('챗봇 모델 선정')
        st.text('')

        
        replicate_api = st.text_input('Replicate API Key:', type='password')
        if replicate_api:
            st.success('API Key 확인 완료!', icon='✅')
        else:
            st.warning('API key를 입력하세요.', icon='⚠️')

        st.text('')
        gpt_type= st.selectbox("🧠  GPT 모델 (LLM) ", ('Llama2-7B','Llama2-13B','Llama2-70B'))
        st.caption('')
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        st.caption('')
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        st.caption('')
        max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
        st.text('')


        # if service_type == '은행':
        #     chatbot_type= st.selectbox("🔊🤖  음성봇 선택", ('완전판매 모니터링',))
        #     st.text('')

        # if service_type == '보험':
        #     chatbot_type= st.selectbox("🔊🤖  음성봇 선택", ('완전판매 모니터링',))
        #     st.text('')

        # if service_type == '카드':
        #     chatbot_type= st.selectbox("🔊🤖  음성봇 선택", ('완전판매 모니터링',))
        #     st.text('')

        # elif service_type == '증권':
        #     chatbot_type= st.selectbox("🔊🤖  음성봇 선택", ('완전판매 모니터링',))
        #     st.text('')
  

        st.subheader('📋 옵션')
        flow_visualize = st.checkbox('📊 시스템 구성도')
        plot_voicegpt = st.checkbox('🔊🤖 챗봇 상담 시작')
        

    if flow_visualize:
        st.divider()
        st.header('📊 시스템 구성도')
        st.header('')

        graph = graphviz.Digraph()
        graph.edge('Ⅰ. 본인확인', 'Ⅰ-⒜ 아니오')
        graph.edge('Ⅰ-⒜ 아니오','상담 종료')
        graph.edge('Ⅰ. 본인확인', 'Ⅰ-⒝ 네')
        graph.edge('Ⅰ-⒝ 네', 'Ⅱ. 통화 가능 여부')
        graph.edge('Ⅱ. 통화 가능 여부', 'Ⅱ-⒜ 아니오')
        graph.edge('Ⅱ-⒜ 아니오', '상담 종료')
        graph.edge('Ⅱ. 통화 가능 여부', 'Ⅱ-⒝ 네')
        graph.edge('Ⅱ-⒝ 네', 'Ⅲ. 자필서명 확인')
        graph.edge('Ⅲ. 자필서명 확인', 'Ⅲ-⒜ 아니오')
        graph.edge('Ⅲ-⒜ 아니오', '서명 입력')
        graph.edge('서명 입력','Ⅲ. 자필서명 확인')
        graph.edge('Ⅲ. 자필서명 확인', 'Ⅲ-⒝ 네')
        graph.edge('Ⅲ-⒝ 네', 'Ⅳ. 내용인지 확인')
        graph.edge('Ⅳ. 내용인지 확인', 'Ⅳ-⒜. 아니오')
        graph.edge('Ⅳ-⒜. 아니오','상담 종료')
        graph.edge('Ⅳ. 내용인지 확인', 'Ⅳ-⒝. 네')
        graph.edge('Ⅳ-⒝. 네', 'Ⅴ. 추가 질문')
        graph.edge('Ⅴ. 추가 질문', 'Ⅴ-⒜ 아니오')
        graph.edge('Ⅴ-⒜ 아니오', '상담 종료')
        graph.edge('Ⅴ. 추가 질문', 'Ⅴ-⒝ 네')
        graph.edge('Ⅴ-⒝ 네', 'Fine-Tune GPT 응답')
        graph.edge('Fine-Tune GPT 응답','Ⅴ. 추가 질문')
        st.graphviz_chart(graph)
        st.subheader('')


    st.divider()
    st.header(f"🔊🤖 {service_type} {chatbot_type} 음성봇")
    st.caption('')





    if plot_voicegpt:
        gpt(user_name,user_date,service_type,replicate_api,gpt_type,temperature,top_p,max_length)



# .venv\Scripts\activate.bat
PJT3()