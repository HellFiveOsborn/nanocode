# Nano Code

CLI de coding assistant local, construída em Rust, dedicada a tiny/small models GGUF.

O Nano Code nasce para um cenário específico: desenvolvimento assistido por IA com custo previsível, privacidade local e execução viável em hardware comum. Em vez de depender de modelos gigantes na nuvem, ele otimiza a experiência para modelos menores, quantizados e eficientes.

## Proposta do produto

- **Local-first de verdade**: inferência local com `llama.cpp` embarcado, sem exigir API externa para funcionar.
- **Foco em tiny models**: seleção de quantização por hardware, priorizando equilíbrio entre qualidade, latência e memória.
- **CLI para fluxo real de engenharia**: TUI interativa, modo programático (`--prompt`) e ferramentas de código integradas.
- **Controle operacional**: permissões por tool (`Always`, `Ask`, `Never`), perfis de agente (`default`, `plan`, `build`/`accept-edits`) e configuração explícita.
- **Extensível com MCP**: descoberta e proxy de tools remotas via servidores MCP (`stdio`, `http` e `streamable-http`).
- **Interação humana no turno**: suporte à tool `ask_user_question` com app inferior dedicado na TUI para coletar decisão/resposta do usuário.
- **Governança de contexto**: middleware no loop (`TurnLimit`, `AutoCompact`, `ContextWarning`, `PlanAgent`) com compactação automática.
- **UX de chat em terminal**: renderização Markdown no chat da assistant (headers, listas, blockquote, fenced code com highlight, tabelas) e fluxo de revisão de plano no Plan Mode.

## Por que tiny models

Modelos menores não são apenas uma escolha "barata"; são uma escolha de produto:

- inicializam mais rápido;
- rodam em máquinas sem GPU high-end;
- reduzem dependência de rede e custo variável por token;
- permitem uso contínuo em ambiente corporativo com requisitos de privacidade.

O trade-off é claro: menos capacidade bruta que modelos de fronteira, exigindo escolhas mais cuidadosas de prompt, contexto e quantização.

## Arquitetura do Nano Code

- `nanocode-cli`: binário `nanocode`, TUI, comandos (`setup`, `sessions`, `config`) e execução interativa/programática.
- `nanocode-core`: loop do agente, integração de tools (`bash`, `read_file`, `write_file`, `grep`, `search_replace`, `task`, `ask_user_question`), configuração, sessões e `SkillManager` (discovery + filtros + injeção no system prompt).
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

Retomar sessão:

```bash
cargo run -p nanocode-cli -- --continue
cargo run -p nanocode-cli -- --resume <uuid-da-sessao>
```

Na saída da TUI, quando houver sessão ativa, o CLI exibe o comando de retomada:

```bash
nanocode --resume <uuid-da-sessao>
```

Dentro da TUI, use:

- `/config` para abrir o app inferior de configurações operacionais (runtime + permissões de tools);
- `/model` ou `/models` para abrir o seletor de modelos;
- `/setup` como alias de `/model`;
- skills locais agora aparecem no menu `/` automaticamente e podem ser invocadas por slash (ex.: `/frontend-design` e `/javascript-backend`);
- skills são descobertas em `.nanocode/skills`, `./skills`, `.agents/skills`, `.claude/skills` e `~/.config/nanocode/skills`;
- as skills distribuídas com o NanoCode são auto-instaladas em `~/.config/nanocode/skills` na inicialização do CLI;
- `/resume` para listar sessões e retomar uma com `↑/↓` + `Enter`;
- `/continue` como alias de `/resume`;
- `/rewind` para desfazer a alteração de arquivo mais recente registrada na sessão (pode ser repetido até esgotar o histórico);
- `/agent` para ver o perfil de agente ativo e os disponíveis;
- `/agent <nome>` para trocar o perfil na sessão atual;
- com input vazio, `Tab` e `Shift+Tab` ciclam os modos primários (`plan` → `build` → `default`);
- `explore` é subagente read-only para delegação via `task` e não entra no ciclo primário de modos;
- no Plan Mode, ao final do plano aparece um modal de decisão:
  - `Sim, mudar para Build`;
  - `Não, continuar em Plan`;
  - `Digitar ajustes para refazer o plano`;
  - `Tab` na opção 3 entra direto no modo de digitação;
  - no `Sim`, o modo `build` reinicia com contexto limpo e recebe apenas o plano aprovado como contexto inicial de execução;
- no seletor, primeiro você escolhe o modelo; depois escolhe a variante de quantização;
- o seletor mostra metadados (Thinking/visão/contexto), recomendação por hardware e status de cache;
- ao trocar a variante de um modelo, o Nano Code remove automaticamente as variantes antigas desse mesmo modelo no cache (mantém apenas uma);
- modal de aprovação de tools com três decisões: permitir uma vez, permitir sempre para esta tool na sessão, negar;
- quando a IA chama `ask_user_question`, a TUI abre o app inferior `question`:
  - `↑/↓` navega opções;
  - `1..9` seleciona opção rápida;
  - texto livre é aceito quando habilitado pela tool;
  - `Enter` confirma e `Esc` cancela;
- compactação automática de contexto quando o threshold configurado é atingido, com feedback no chat;
- `Esc` para fechar `/config` (salva mudanças) ou voltar entre telas no seletor de modelos.

Entrada/paste no composer:

- `Shift+Enter`/`Ctrl+Enter` inserem nova linha;
- `@arquivo` ou `@pasta|busca` anexam contexto direto no prompt (com autocomplete via `Tab`, `↑/↓` para navegação);
- colagem textual multilinha/larga usa resumo automático no input (`[Pasted ~N lines #X]`) e envia o conteúdo completo ao modelo no submit;
- `Ctrl+V` tenta colar imagem da área de transferência e cria placeholder (`[Image X]`) com envio como `image_url` (data URL) para modelos multimodais;
- colagem de imagem é bloqueada automaticamente quando o modelo ativo não suporta visão.

Observação: o catálogo atual está com `Qwen3 4B Thinking` (`vision: não`), então `Ctrl+V` com imagem ficará bloqueado até selecionar um modelo com visão.

Configuração MCP (em `~/.config/nanocode/config.toml`):

```toml
[[mcp_servers]]
transport = "stdio"
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]

[[mcp_servers]]
transport = "streamable-http"
name = "remote"
url = "https://example.com/mcp"
headers = { "X-API-Key" = "token" }
```

As tools descobertas entram automaticamente no runtime com prefixo `mcp_<servidor>_<tool>`.

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
