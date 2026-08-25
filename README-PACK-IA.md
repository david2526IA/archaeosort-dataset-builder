# Pack Portatil De Instrucciones Para IA

Este directorio contiene un kit listo para copiar al raiz de cualquier proyecto
para que asistentes de codigo y modelos de IA trabajen con menos tokens, menos
ruido y mas eficacia.

Importante:
- Ningun archivo puede forzar al 100% a todas las IAs del mercado.
- Este pack maximiza compatibilidad usando nombres y rutas que varios asistentes
  suelen leer automaticamente.
- Si una herramienta no respeta archivos del repo, usa `PROMPT-MANUAL.md`.

## Como usarlo

1. Copia TODO el contenido de `md/` al raiz del proyecto destino.
2. No copies la carpeta `md/` entera dentro del proyecto.
3. Asegurate de incluir tambien las carpetas ocultas:
   `.github`, `.cursor`, `.continue`, `.windsurf`.
4. Abre el proyecto en tu herramienta de IA.
5. Pide a la IA: `Resume las reglas locales que estas siguiendo en este repo`.

## Copia rapida en PowerShell

```powershell
Get-ChildItem -Force .\md | Copy-Item -Destination C:\ruta\tu-proyecto -Recurse -Force
```

## Instalador para Windows

Si aun tienes este repositorio fuente a mano, tambien puedes usar el asistente:

```powershell
.\instalar-pack.bat
```

O pasar la ruta destino directamente:

```powershell
.\instalar-pack.bat "C:\ruta\tu-proyecto"
```

Modo sin preguntas:

```powershell
.\instalar-pack.bat "C:\ruta\tu-proyecto" /Y
```

## Instalador para Ubuntu o Linux

Si aun tienes este repositorio fuente a mano:

```bash
chmod +x ./instalar-pack.sh
./instalar-pack.sh
```

O pasando la ruta destino:

```bash
./instalar-pack.sh /ruta/tu-proyecto
```

Modo sin preguntas:

```bash
./instalar-pack.sh /ruta/tu-proyecto -y
```

## Archivos incluidos

- `REGLAS-IA-PORTABLES.md`
  Regla canonica, neutral y reutilizable.
- `PROMPT-MANUAL.md`
  Texto corto para pegar en "custom instructions" o en el primer mensaje.
- `AGENTS.md`
  Instrucciones para asistentes que leen `AGENTS.md`.
  Es la pieza principal para Codex.
- `CODEX.md`
  Refuerzo explicito y legible para sesiones con Codex.
- `CODEX-PROMPT.md`
  Prompt breve para pegar manualmente en Codex si hiciera falta.
- `CLAUDE.md`
  Variante principal para Claude Code.
- `CLAUDE.local.md`
  Refuerzo adicional para Claude Code cuando tambien se lean overrides locales.
- `CLAUDE-CODE-PROMPT.md`
  Prompt breve para pegar manualmente en Claude Code si hiciera falta.
- `GEMINI.md`
  Variante simple para herramientas que usen ese nombre.
- `CLINE.md`
  Variante simple para herramientas que no tengan un formato mejor.
- `.github/copilot-instructions.md`
  Instrucciones para GitHub Copilot.
- `.cursor/rules/00-portable-core.mdc`
  Regla global para Cursor.
- `.continue/rules/00-portable-core.md`
  Regla para Continue.
- `.windsurf/rules/00-portable-core.md`
  Regla para Windsurf.

## Resultado esperado

Cuando una IA lea estos archivos deberia:
- inspeccionar antes de tocar codigo
- leer solo lo relevante
- responder con brevedad
- aplicar cambios minimos y verificables
- ahorrar tokens sin perder calidad
- no romper el proyecto por prisas ni por respuestas largas

## Compatibilidad por herramienta

- Codex:
  usa sobre todo `AGENTS.md`. Este pack añade tambien `CODEX.md` y
  `CODEX-PROMPT.md` como refuerzo explicito.
- Claude Code:
  usa sobre todo `CLAUDE.md`. Este pack añade tambien `CLAUDE.local.md` y
  `CLAUDE-CODE-PROMPT.md` como respaldo adicional.
- Copilot:
  usa `.github/copilot-instructions.md`.
- Cursor:
  usa `.cursor/rules/*.mdc`.
- Continue:
  usa `.continue/rules/*.md`.
- Windsurf:
  usa `.windsurf/rules/*.md`.

## Limite real

Si una extension ignora archivos locales y solo usa instrucciones de cuenta o de
sesion, este pack no puede obligarla por si solo. En ese caso:
- pega `PROMPT-MANUAL.md` o el prompt especifico de la herramienta
- o envialo como primer mensaje en cada sesion
