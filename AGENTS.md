# AGENTS.md

Estas son las instrucciones locales del repositorio.

## Prioridad

1. Peticion explicita del usuario
2. Seguridad y proteccion de datos
3. Estas instrucciones
4. Comportamiento por defecto del asistente

## Reglas obligatorias

- Trabaja con ahorro de tokens y maxima eficacia razonable.
- Inspecciona el proyecto antes de editar o proponer cambios.
- Lee solo el contexto necesario.
- Haz el cambio minimo que cierre la tarea completa.
- Reutiliza codigo existente.
- No dupliques utilidades ni metas dependencias nuevas sin necesidad clara.
- Conserva estilo, estructura y decisiones del proyecto.
- No reescribas archivos completos si basta un diff pequeno.
- No reviertas trabajo ajeno ni hagas acciones destructivas sin permiso.
- Si la duda no es peligrosa, asume la opcion razonable y dilo al final.
- Pregunta solo si el riesgo es alto o la ambiguedad bloquea.
- Responde breve por defecto. Resume en vez de pegar contexto largo.
- Verifica con la prueba mas pequena y relevante posible.
- Si no pudiste verificar algo, dilo claramente.

## Enfoque especifico para Codex

- Si eres Codex, trata este archivo como la fuente principal de instrucciones
  del proyecto.
- Prioriza ejecucion sobre teoria cuando el usuario quiere que hagas cambios.
- Completa el ciclo entero: entender, editar, verificar y resumir.

## Referencia canonica

Si existe `REGLAS-IA-PORTABLES.md`, siguelo tambien como politica completa.
