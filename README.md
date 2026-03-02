# Nano Code

CLI de coding assistant local, construída em Rust, dedicada a tiny/small models GGUF.

O Nano Code nasce para um cenário específico: desenvolvimento assistido por IA com custo previsível, privacidade local e execução viável em hardware comum. Em vez de depender de modelos gigantes na nuvem, ele otimiza a experiência para modelos menores, quantizados e eficientes.

## Proposta do produto

- **Local-first de verdade**: inferência local com `llama.cpp` embarcado, sem exigir API externa para funcionar.
- **Foco em tiny models**: seleção de quantização por hardware, priorizando equilíbrio entre qualidade, latência e memória.
- **CLI para fluxo real de engenharia**: TUI interativa, modo programático (`--prompt`) e ferramentas de código integradas.
- **Controle operacional**: permissões por tool (`Always`, `Ask`, `Never`), perfis de agente (`default`, `plan`, `accept-edits`, `auto-approve`) e configuração explícita.
- **Governança de contexto**: middleware no loop (`TurnLimit`, `AutoCompact`, `ContextWarning`, `PlanAgent`) com compactação automática.

## Por que tiny models

Modelos menores não são apenas uma escolha "barata"; são uma escolha de produto:

- inicializam mais rápido;
- rodam em máquinas sem GPU high-end;
- reduzem dependência de rede e custo variável por token;
- permitem uso contínuo em ambiente corporativo com requisitos de privacidade.

O trade-off é claro: menos capacidade bruta que modelos de fronteira, exigindo escolhas mais cuidadosas de prompt, contexto e quantização.

## Arquitetura do Nano Code

- `nanocode-cli`: binário `nanocode`, TUI, comandos (`setup`, `sessions`, `config`) e execução interativa/programática.
- `nanocode-core`: loop do agente, integração de tools (`bash`, `read_file`, `write_file`, `grep`, `search_replace`), configuração e sessão.
- `nanocode-hf`: detecção de hardware, catálogo de quantizações e download de modelos via Hugging Face.

## Modelo atual e estratégia de quantização

Hoje o projeto opera com um modelo principal:

- **Qwen3 4B Thinking** (`unsloth/Qwen3-4B-Thinking-2507-GGUF`)
  - categoria: `Thinking`
  - visão: `não`
  - contexto máximo: `262144`

Faixa de quantizações suportada no setup:

- de `Q2_K` (~1.6 GB) até `F16` (~8.5 GB), com recomendação automática por memória disponível.

Política atual de recomendação (hardware):

- `>= 16 GB`: `Q5_K_M`
- `>= 10 GB`: `Q4_K_M`
- `>= 6 GB`: `Q3_K_M`
- `>= 4 GB`: `Q2_K`
- `< 4 GB`: `Q2_K`

## Trade-offs assumidos pelo produto

- **Ganha**: autonomia local, previsibilidade de custo, menor fricção para rodar no dia a dia.
- **Aceita**: menor robustez em tarefas muito longas/complexas comparado a modelos maiores.
- **Compensa**: ajuste fino de quantização, contexto e KV cache (`--ctk` / `--ctv`) para extrair o máximo do hardware.

## Posicionamento

Nano Code não tenta ser "o agente mais poderoso da nuvem".  
Nano Code quer ser o **assistente de código local mais prático para tiny models**.

Se o objetivo for produtividade local com controle técnico, ele é o produto certo.

## Uso rápido

Primeiro setup (detecção de hardware + escolha/download de quantização):

```bash
cargo run -p nanocode-cli -- setup
```

Modo interativo (TUI):

```bash
cargo run -p nanocode-cli --
```

Dentro da TUI, use:

- `/config` para abrir a tela de configurações operacionais (runtime + permissões de tools);
- `/model` ou `/models` para abrir o seletor de modelos;
- `/setup` como alias de `/model`;
- `/agent` para ver o perfil de agente ativo e os disponíveis;
- `/agent <nome>` para trocar o perfil na sessão atual;
- no seletor, primeiro você escolhe o modelo; depois escolhe a variante de quantização;
- o seletor mostra metadados (Thinking/visão/contexto), recomendação por hardware e status de cache;
- ao trocar a variante de um modelo, o Nano Code remove automaticamente as variantes antigas desse mesmo modelo no cache (mantém apenas uma);
- modal de aprovação de tools com três decisões: permitir uma vez, permitir sempre para esta tool na sessão, negar;
- compactação automática de contexto quando o threshold configurado é atingido, com feedback no chat;
- `Esc` para fechar `/config` (salva mudanças) ou voltar entre telas no seletor de modelos.

Modo programático:

```bash
cargo run -p nanocode-cli -- --prompt "Revise este arquivo"
```

Modo programático com perfil de agente:

```bash
cargo run -p nanocode-cli -- --agent plan --prompt "Mapeie a estrutura deste projeto"
```

Comandos úteis:

```bash
cargo run -p nanocode-cli -- sessions
cargo run -p nanocode-cli -- config
```

Build de release:

```bash
cargo build --release -p nanocode-cli
```

Binário:

```bash
target/release/nanocode
```

## Licença

Apache-2.0.
