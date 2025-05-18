# Célula 1 (Integrada): Instalação das bibliotecas necessárias
print("Instalando bibliotecas necessárias (CrewAI, LiteLLM, Langchain Community, Flask, Twilio, PyNgrok)...")
!pip install -q crewai crewai-tools langchain-community Flask twilio pyngrok python-dotenv

print("Bibliotecas instaladas.")

# Célula 2 (Integrada): Configuração da API Keys e Inicialização da LLM
import os
import threading
import time
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client # Importar o cliente Twilio
from google.colab import userdata
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatLiteLLM
from pyngrok import ngrok
from werkzeug.serving import run_simple

# --- Configuração das API Keys do Google Gemini ---
print("\nConfigurando API Key do Google Gemini...")
try:
    google_api_key_value = userdata.get('GOOGLE_API_KEY')
    if not google_api_key_value:
        raise ValueError("GOOGLE_API_KEY não encontrada nos Secrets do Colab.")
    os.environ["GOOGLE_API_KEY"] = google_api_key_value
    print("✅ API Key do Google Gemini carregada com sucesso do Colab Secrets.")
except Exception as e:
    print(f"❌ ERRO FATAL: Falha ao carregar API Key do Google Gemini: {e}")
    print("Por favor, verifique sua 'GOOGLE_API_KEY' nos Colab Secrets.")
    exit()

# --- Configuração das API Keys da Twilio ---
print("\nConfigurando API Keys da Twilio...")
try:
    twilio_account_sid_value = userdata.get('TWILIO_ACCOUNT_SID')
    twilio_auth_token_value = userdata.get('TWILIO_AUTH_TOKEN')
    twilio_phone_number_value = userdata.get('TWILIO_PHONE_NUMBER') # NOVO: seu número da Twilio

    if not all([twilio_account_sid_value, twilio_auth_token_value, twilio_phone_number_value]):
        raise ValueError("Uma ou mais credenciais da Twilio não encontradas nos Secrets do Colab.")

    os.environ["TWILIO_ACCOUNT_SID"] = twilio_account_sid_value
    os.environ["TWILIO_AUTH_TOKEN"] = twilio_auth_token_value
    os.environ["TWILIO_PHONE_NUMBER"] = twilio_phone_number_value
    print("✅ API Keys e número de telefone da Twilio carregados com sucesso do Colab Secrets.")

except Exception as e:
    print(f"❌ ERRO FATAL: Falha ao carregar API Keys da Twilio: {e}")
    print("Por favor, adicione 'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN' e 'TWILIO_PHONE_NUMBER' nos Colab Secrets.")
    exit()

# Inicializa o modelo Gemini LLM usando ChatLiteLLM
gemini_llm = ChatLiteLLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

# Inicializa o cliente Twilio
twilio_client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])


print("✅ Modelo Gemini LLM e Cliente Twilio configurados com sucesso.")


# --- DEFINIÇÃO DOS AGENTES PARA O FLUXO DE ASSISTÊNCIA AO PROJETO DO USUÁRIO ---
pesquisador_mercado = Agent(
    role='Especialista em Pesquisa e Tendências de Mercado',
    goal='Investigar tendências, tecnologias e soluções existentes relevantes para a demanda de projeto do usuário, fornecendo insights para o debate.',
    backstory='Você é um pesquisador incansável, sempre em busca de informações atualizadas para embasar decisões de projeto e identificar oportunidades.',
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

estrategista_tecnico = Agent(
    role='Estrategista de Soluções e Facilitador de Debate',
    goal='Liderar o debate para conceituar a melhor abordagem técnica e funcional para o projeto do usuário, considerando viabilidade, inovação e escalabilidade.',
    backstory='Com vasta experiência em arquitetura de sistemas e metodologias ágeis, você facilita a discussão e direciona a equipe para soluções eficazes e criativas.',
    llm=gemini_llm,
    verbose=True,
    allow_delegation=True
)

consolidor_de_prompt = Agent(
    role='Engenheiro de Prompt e Consolidador de Requisitos',
    goal='Transformar todas as discussões, pesquisas e debates em um prompt técnico claro, conciso e completo, pronto para ser validado pela equipe externa.',
    backstory='Você é um mestre na arte de resumir e estruturar informações complexas em documentos técnicos de alta qualidade e prompts acionáveis, garantindo que nada essencial seja perdido.',
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

validador_de_prompt_externo = Agent(
    role='Analista de Validação Externa e Qualidade de Prompt',
    goal='Validar a qualidade, clareza e completude do prompt técnico gerado, garantindo que ele atenda à demanda original do usuário e esteja pronto para a equipe de execução.',
    backstory='Seu foco é a qualidade e a experiência do usuário final. Você garante que o prompt seja compreensível, útil e alinhado com as expectativas do cliente para quem vai executá-lo, identificando falhas e sugerindo aprimoramentos.',
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

print("\nAgentes para Assistência de Projetos (fluxo do usuário) definidos.")


# --- DEFINIÇÃO DAS TAREFAS PARA O FLUXO DE ASSISTÊNCIA AO PROJETO DO USUÁRIO ---
pesquisar_demanda_task = Task(
    description="{demanda_usuario} - Pesquisar e coletar informações relevantes sobre a demanda do usuário. Identifique os requisitos funcionais e não-funcionais, tecnologias mencionadas e desafios potenciais. Prepare um resumo para o debate.",
    expected_output="Um resumo detalhado da demanda do usuário, incluindo pontos chave, tecnologias, escopo inicial e quaisquer incertezas a serem discutidas.",
    agent=pesquisador_mercado
)

debater_e_conceituar_task = Task(
    description="Com base na pesquisa da demanda e em conhecimentos técnicos, debata as melhores abordagens e soluções técnicas para o projeto do usuário. O objetivo é conceituar a estrutura do projeto, tecnologias principais e um plano de alto nível, considerando viabilidade, inovação e escalabilidade.",
    expected_output="Um rascunho de plano de projeto de alto nível, com a estrutura da solução, tecnologias principais debatidas e possíveis alternativas.",
    agent=estrategista_tecnico
)

consolidar_em_prompt_task = Task(
    description="Consolidar os resultados da pesquisa e do debate em um *prompt técnico detalhado*. Este prompt deve ser um guia claro e acionável para a equipe de execução, incluindo: visão geral do projeto, requisitos funcionais, requisitos técnicos, tecnologias sugeridas, e a estrutura de módulos/componentes.",
    expected_output="Um prompt técnico completo e bem estruturado, pronto para ser validado, contendo todos os detalhes essenciais para iniciar o desenvolvimento (formato Markdown).",
    agent=consolidor_de_prompt
)

validar_e_apresentar_prompt_task = Task(
    description="Revise o prompt técnico final gerado a partir da demanda '{demanda_usuario}'. Valide sua clareza, completude, alinhamento com a demanda original do usuário, e se ele está pronto para ser entregue à equipe de execução. Formate a saída para uma apresentação amigável e concisa ao usuário do WhatsApp, incluindo um resumo do plano e indicando que ele foi validado, além de mencionar os próximos passos (que a equipe de execução vai trabalhar nisso). Se houver correções, inclua-as de forma clara. Mantenha a resposta concisa para WhatsApp.",
    expected_output="O prompt técnico final validado ou um relatório conciso com sugestões de correção. O output deve ser direto para o usuário do WhatsApp, com um resumo do plano e os próximos passos claros.",
    agent=validador_de_prompt_externo
)

print("\nTarefas para Assistência de Projetos (fluxo do usuário) definidas.")


app = Flask(__name__)

# --- FUNÇÃO ASSÍNCRONA PARA PROCESSAR CREWAI E ENVIAR RESULTADO FINAL ---
def send_crew_result_async(user_message: str, sender_number: str):
    """
    Executa o processo CrewAI em background e envia a mensagem final para o usuário.
    """
    print(f"DEBUG: Início do processamento CrewAI assíncrono para '{user_message}'")
    try:
        crew_brainstorming_e_validacao = Crew(
            agents=[
                pesquisador_mercado,
                estrategista_tecnico,
                consolidor_de_prompt,
                validador_de_prompt_externo
            ],
            tasks=[
                pesquisar_demanda_task,
                debater_e_conceituar_task,
                consolidar_em_prompt_task,
                validar_e_apresentar_prompt_task
            ],
            process=Process.sequential,
            manager_llm=gemini_llm,
            llm=gemini_llm,
            verbose=True
        )

        resultado_do_prompt_tecnico = crew_brainstorming_e_validacao.kickoff(inputs={
            'demanda_usuario': user_message
        })
        final_message = resultado_do_prompt_tecnico
        print(f"DEBUG: Resultado final do prompt técnico gerado para o usuário: \n{final_message}")

    except Exception as crew_error:
        print(f"❌ Erro ao executar a Crew de Brainstorming assíncrona: {crew_error}")
        final_message = "Desculpe, nossa equipe de IA teve um problema ao gerar o plano do seu projeto. Por favor, tente novamente com uma descrição um pouco diferente, ou entre em contato com o suporte."

    # Envia a mensagem final para o usuário usando o cliente Twilio
    try:
        twilio_client.messages.create(
            from_=os.environ["TWILIO_PHONE_NUMBER"], # Seu número Twilio habilitado para WhatsApp
            to=sender_number,
            body=final_message
        )
        print(f"DEBUG: Mensagem final do CrewAI enviada para {sender_number}.")
    except Exception as e:
        print(f"❌ ERRO ao enviar mensagem final do CrewAI para {sender_number}: {e}")

# --- FUNÇÃO SÍNCRONA PARA RESPOSTAS IMEDIATAS (NÃO CREWAI) ---
def get_immediate_response(user_message: str) -> str:
    """
    Retorna a resposta imediata para mensagens que não exigem o processo CrewAI.
    """
    user_message_lower = user_message.lower().strip()
    message_word_count = len(user_message.split())

    # --- NÍVEL 1: SAUDAÇÕES MUITO CURTAS OU COMANDOS INICIAIS EXATOS ---
    greetings = ["ola", "olá", "oi", "bom dia", "boa tarde", "boa noite", "hi"]
    start_commands_exact = ["começar", "iniciar projeto", "novo projeto", "criar projeto"]
    help_commands_exact = ["ajuda", "suporte", "dúvida", "duvida"]

    if message_word_count <= 3 and any(cmd == user_message_lower for cmd in greetings + start_commands_exact + help_commands_exact):
        if any(cmd == user_message_lower for cmd in help_commands_exact):
            return (
                "Claro! Sou seu Assistente de Projetos com IA. Por favor, me diga em que posso te ajudar com mais detalhes. "
                "Seja para planejar um novo projeto, tirar uma dúvida sobre algo que já geramos, ou para resolver um problema técnico."
            )
        else:
            return (
                "Olá! Sou seu **Assistente de Projetos com IA**. Nossa equipe de especialistas está pronta para te ajudar a conceituar e planejar seu projeto.\n\n"
                "**Como posso te ajudar hoje?** Por favor, me diga qual a sua ideia de projeto, o problema que você quer resolver, ou se precisa de suporte com algo específico."
            )

    # --- NÍVEL 2: DISTINGUIR ENTRE SUPORTE ESPECÍFICO OU PEDIDO DE MAIS DETALHES ---
    support_keywords = ["erro", "bug", "funciona", "problema no código", "executar", "link", "tutorial", "documentação", "corrigir", "implementar", "instalar"]
    
    if message_word_count < 15 and any(kw in user_message_lower for kw in support_keywords):
        print("DEBUG: Mensagem parece ser uma pergunta de suporte específica. Encaminhando para o LLM direto para resposta de suporte.")
        suporte_response = gemini_llm.invoke(f"Como Assistente de Projetos com IA, o usuário perguntou: '{user_message}'. Responda de forma concisa e útil, oferecendo ajuda na execução, correção ou fornecendo recursos (links, documentação, tutoriais), se aplicável. O usuário está interagindo via WhatsApp, então a resposta deve ser direta e em português. Lembre-se que você é um consultor de execução de projetos.")
        return suporte_response.content if hasattr(suporte_response, 'content') else str(suporte_response)
    
    # Se nenhuma das condições acima for atendida, é uma ideia de projeto curta/ambígua que precisa de mais detalhes
    print("DEBUG: Mensagem parece ser uma ideia de projeto curta ou ambígua. Solicitando mais detalhes.")
    return (
        "Entendi sua ideia! Para que nossa equipe de especialistas possa criar um plano robusto, preciso de mais detalhes.\n\n"
        "**Vamos lá detalhar o problema/ideia:** Por favor, me conte mais sobre:\n"
        "**1. Qual o problema principal que seu projeto resolve ou a ideia central?**\n"
        "**2. Quais os objetivos? O que ele deve fazer ou entregar?**\n"
        "**3. Já pensou em alguma tecnologia (ex: Python, React, mobile)?**\n"
        "**4. Há alguma restrição importante (prazo, orçamento, privacidade)?**\n\n"
        "Quanto mais detalhes, melhor! Assim podemos criar um prompt técnico mais acertivo."
    )


@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '').strip()

    print(f"Mensagem recebida de {sender_number}: {incoming_msg}")

    resp = MessagingResponse()
    
    message_word_count = len(incoming_msg.split())

    # Decide se a mensagem é um projeto detalhado que exige CrewAI (processo longo)
    # ou uma mensagem que pode ser respondida imediatamente.
    if message_word_count >= 15: # O limite de 15 palavras pode ser ajustado
        # Envia a mensagem de "Aguarde" imediatamente
        resp.message("Aguarde um instante, por favor! Nossos especialistas de IA estão analisando sua demanda e debatendo a melhor abordagem. Isso pode levar alguns minutos. Assim que tivermos uma resposta ou o plano inicial, te avisaremos! 😊")
        
        # Inicia uma nova thread para processar a tarefa longa e enviar a mensagem final
        thread = threading.Thread(target=send_crew_result_async, args=(incoming_msg, sender_number))
        thread.daemon = True # Permite que o programa principal saia mesmo se a thread ainda estiver rodando
        thread.start()
    else:
        # Para mensagens curtas/simples, obtém a resposta imediatamente e a envia
        immediate_response = get_immediate_response(incoming_msg)
        resp.message(immediate_response)

    return str(resp)

# Função para rodar o servidor Flask em uma thread
def run_flask_app_thread():
    run_simple('0.0.0.0', 5000, app, use_reloader=False, use_debugger=False)

# Função para iniciar ngrok e obter a URL pública
def start_ngrok_tunnel(port):
    print("Iniciando túnel Ngrok...")
    try:
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        print(f"🚀 Ngrok Tunnel URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Erro ao iniciar Ngrok: {e}")
        print("Certifique-se de ter autenticado o Ngrok. Execute '!ngrok config add-authtoken SEU_TOKEN_NGROK' em uma célula separada.")
        return None

if __name__ == "__main__":
    # --- PROMPT E AGENTES QUE "CONSTROEM" O PRÓPRIO BOT DE ASSISTÊNCIA DE PROJETOS (RODA APENAS UMA VEZ) ---
    engenheiro_requisitos = Agent(
        role='Engenheiro de Requisitos de Software',
        goal='Traduzir requisitos do projeto do bot em funcionalidades e especificações claras.',
        backstory='Você é um engenheiro de requisitos experiente em transformar conceitos de bots de IA em especificações técnicas detalhadas.',
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False
    )

    arquiteto_software = Agent(
        role='Arquiteto de Software e IA',
        goal='Definir a arquitetura técnica do bot, suas interações com LLMs e APIs.',
        backstory='Com anos de experiência em engenharia de software e IA, você estrutura soluções escaláveis e eficientes para bots conversacionais.',
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False
    )

    dev_backend = Agent(
        role='Desenvolvedor Python Back-End do Bot',
        goal='Desenvolver a lógica de comunicação, orquestração de Crews e integração com APIs da Twilio e Google Gemini.',
        backstory='Você é um desenvolvedor Python focado na criação de APIs robustas e lógica de negócio para bots de IA.',
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False
    )

    dev_frontend = Agent(
        role='Desenvolvedor Front-End de Interface de Teste',
        goal='Criar interfaces de teste e mensagens iniciais amigáveis para o bot.',
        backstory='Especialista em experiências digitais, você transforma funcionalidades em interfaces bonitas e intuitivas, e mensagens claras para o usuário.',
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False
    )

    qa_validador = Agent(
        role='Validador QA do Bot',
        goal='Validar se o bot atende aos requisitos de funcionalidade, privacidade e usabilidade.',
        backstory='Você é um analista de qualidade com olhar atento para erros e inconsistências em bots de IA.',
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False
    )
    
    suporte_usuario = Agent(
        role='Documentador e Suporte de Conhecimento do Bot',
        goal='Documentar as funcionalidades do bot e preparar guias de uso/solução de problemas internos.',
        backstory='Você é essencial para garantir que o conhecimento sobre o bot esteja acessível e que ele possa ser mantido e aprimorado.',
        llm=gemini_llm,
        verbose=False,
        allow_delegation=False
    )

    # Tarefas genéricas para desenvolvimento de software (agora descrevendo o bot em si)
    analisar_requisitos_task = Task(
        description='Analisar o prompt que define o "Assistente de Projetos com IA para WhatsApp" e extrair seus requisitos, funcionalidades e fluxo de interação.',
        expected_output='Lista detalhada de requisitos e funcionalidades para o "Assistente de Projetos com IA para WhatsApp".',
        agent=engenheiro_requisitos
    )

    desenhar_arquitetura_task = Task(
        description='Propor a arquitetura do "Assistente de Projetos com IA para WhatsApp", incluindo tecnologias, camadas da aplicação e como a CrewAI se integrará com Twilio e Gemini.',
        expected_output='Documento de arquitetura detalhado para o "Assistente de Projetos com IA para WhatsApp".',
        agent=arquiteto_software
    )

    implementar_backend_task = Task(
        description='Desenvolver o back-end do "Assistente de Projetos com IA para WhatsApp" conforme os requisitos e arquitetura definidos, focando na orquestração da CrewAI e na API do WhatsApp.',
        expected_output='Código-fonte completo do back-end para o "Assistente de Projetos com IA para WhatsApp".',
        agent=dev_backend
    )

    implementar_frontend_task = Task(
        description='Criar as mensagens de boas-vindas e orientação para o "Assistente de Projetos com IA para WhatsApp" e definir uma interface de teste conceitual se aplicável.',
        expected_output='Textos das mensagens de interação do bot e plano conceitual para interface de teste.',
        agent=dev_frontend
    )

    validar_aplicacao_task = Task(
        description='Testar o "Assistente de Projetos com IA para WhatsApp" para garantir que suas funcionalidades (brainstorming, validação, suporte) operem conforme o esperado e que a privacidade seja mantida.',
        expected_output='Relatório de validação detalhado para o "Assistente de Projetos com IA para WhatsApp".',
        agent=qa_validador
    )

    canal_duvidas_usuario_task_bot = Task(
        description='Compilar FAQ e documentação interna para o uso e manutenção do "Assistente de Projetos com IA para WhatsApp".',
        expected_output='Documentação interna e FAQ para o "Assistente de Projetos com IA para WhatsApp".',
        agent=suporte_usuario
    )

    prompt_aprovado_bot = """
    O projeto a ser desenvolvido é um **"Assistente de Projetos com IA para Profissionais no WhatsApp"**. Este bot tem como objetivo principal ajudar profissionais a conceituar e planejar seus projetos de software ou IA, passando por um processo colaborativo e validado.

    **Fluxo de Interação do Bot com o Usuário (Capacidades Principais):**
    1.  **Brainstorming & Planejamento:** O bot recebe a descrição do projeto do usuário. Uma equipe interna de IA (CrewAI) composta por um *Pesquisador*, *Estrategista* e *Consolidor* debate e gera um "Prompt Técnico Validado" detalhado para o projeto do usuário.
    2.  **Validação Externa:** O "Prompt Técnico Validado" é então analisado por um *Validador Externo* que garante sua clareza, completude e alinhamento com a demanda original do usuário, antes de ser apresentado.
    3.  **Simulação de Execução:** O bot informa ao usuário que o "Prompt Técnico Validado" seria então entregue a uma "Equipe de Execução" (conceitual neste MVP) que desenvolveria back-end, front-end e faria a validação.
    4.  **Canal Aberto para Dúvidas e Suporte:** Após a entrega do prompt, o bot oferece suporte contínuo através de um agente especializado para tirar dúvidas, ajudar na execução local (se o código for gerado), corrigir erros e fornecer recursos (links, docs).

    **Requisitos Chave do Bot (como sistema):**
    1.  **Interface WhatsApp:** Receber e enviar mensagens via Twilio.
    2.  **Orquestração de IA:** Utilizar CrewAI para gerenciar a colaboração entre agentes para planejamento de projetos.
    3.  **Geração de Prompt Técnico:** A saída principal do bot deve ser um prompt técnico detalhado e validado para o projeto do usuário.
    4.  **Suporte Interativo:** Capacidade de responder a perguntas de acompanhamento sobre o plano ou execução.
    5.  **Privacidade Total:** Nenhuma mensagem ou dado do usuário será armazenado em disco. O processamento é em memória.

    **Tecnologias Esperadas para o Bot:** Python (Flask), Google Gemini API (via LiteLLM), CrewAI, Twilio WhatsApp API.
    """

    def create_execution_crew_for_bot_itself(prompt_aprovado: str):
        crew = Crew(
            agents=[
                engenheiro_requisitos,
                arquiteto_software,
                dev_backend,
                dev_frontend,
                qa_validador,
                suporte_usuario
            ],
            tasks=[
                analisar_requisitos_task,
                desenhar_arquitetura_task,
                implementar_backend_task,
                implementar_frontend_task,
                validar_aplicacao_task,
                canal_duvidas_usuario_task_bot
            ],
            process=Process.sequential,
            verbose=False,
            manager_llm=gemini_llm,
            llm=gemini_llm
        )
        print("\n--- Iniciando equipe de execução para 'construir' o próprio Bot Assistente de Projetos (processo inicial) ---")
        return crew.kickoff(inputs={"prompt_aprovado": prompt_aprovado})

    resultado_execucao_bot_proprio = create_execution_crew_for_bot_itself(prompt_aprovado=prompt_aprovado_bot)
    print("\n✅ Relatório de desenvolvimento do Bot Assistente de Projetos (gerado na inicialização):\n")

    flask_thread = threading.Thread(target=run_flask_app_thread)
    flask_thread.daemon = True
    flask_thread.start()

    time.sleep(3)

    ngrok_url = None

    try:
        ngrok_url = start_ngrok_tunnel(5000)

        if ngrok_url:
            print(f"\n✨ Bot de Assistente de Projetos WhatsApp pronto em: {ngrok_url}/whatsapp")
            print("➡️ Configure este URL no seu Twilio WhatsApp Sandbox (Webhook 'WHEN A MESSAGE COMES IN').")
            print("\n**Mensagem de Boas-Vindas e Orientações do Bot para o WhatsApp:**")
            print("Olá! Sou seu Assistente de Projetos com IA. Nossa equipe de especialistas está pronta para ajudar a desenvolver o plano para seu projeto.")
            print("Para começarmos, por favor, descreva em detalhes seu projeto ou problema. Inclua:")
            print("1. A ideia principal / o que você quer criar.")
            print("2. Os objetivos e o que o projeto deve fazer.")
            print("3. Se há alguma tecnologia ou plataforma específica em mente.")
            print("4. Quaisquer restrições ou requisitos importantes (ex: prazo, orçamento, privacidade).")
            print("\nQuando estiver pronto, envie 'iniciar projeto' ou comece direto com a descrição do seu problema/projeto. Se tiver dúvidas *depois* que o plano for gerado, pode perguntar diretamente!")
            print("\n---")
            print("Pressione Enter para encerrar o túnel Ngrok e o servidor Flask.")
            input()
        else:
            print("Não foi possível obter o URL do Ngrok. Verifique os logs acima para erros.")
    finally:
        if ngrok_url:
            ngrok.kill()
            print("Túnel Ngrok e servidor Flask encerrados.")
        else:
            print("Ngrok não foi iniciado, então não há túnel para encerrar.")

    print("\nProcesso concluído. O bot está offline.")