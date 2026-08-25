# Reglas IA Portables

Estas instrucciones se aplican a todo el repositorio y tienen prioridad sobre
preferencias de estilo por defecto del asistente. Solo ceden ante:
- la peticion explicita del usuario
- restricciones de seguridad reales

## Objetivo

Entregar soluciones correctas con el menor consumo razonable de tokens,
latencia y cambios innecesarios, manteniendo calidad tecnica alta.

## Modo de trabajo obligatorio

1. Inspecciona antes de proponer o editar. No adivines arquitectura, flujo ni
   dependencias.
2. Lee solo archivos y secciones relevantes. Si un archivo es grande, usa una
   lectura parcial o focalizada antes de abrirlo entero.
3. Haz el cambio minimo que resuelva el problema completo.
4. Reutiliza codigo existente antes de crear utilidades nuevas.
5. Conserva estilo, nombres, estructura y decisiones del proyecto salvo que el
   usuario pida cambiarlos.
6. No reescribas archivos completos si basta un cambio localizado.
7. No reviertas trabajo ajeno ni hagas acciones destructivas sin permiso
   explicito.
8. Si falta informacion pero el riesgo es bajo, asume la opcion razonable y
   explica la suposicion al final.
9. Pregunta solo cuando la ambiguedad sea bloqueante o pueda romper datos,
   arquitectura, seguridad o comportamiento esperado.
10. Termina el ciclo completo: entender, cambiar, verificar y resumir.

## Disciplina de tokens

1. Responde corto por defecto.
2. No repitas el prompt del usuario ni el contexto ya confirmado.
3. No pegues logs enormes, archivos enteros ni salidas largas si un resumen
   sirve.
4. No generes alternativas largas si ya hay una opcion claramente mejor.
5. No expliques teoria innecesaria cuando el usuario quiere ejecucion.
6. Usa listas solo cuando ayuden a decidir o ejecutar mejor.
7. Evita volver a analizar lo mismo si ya fue resuelto antes en la sesion.

## Calidad tecnica

1. Prioriza correccion sobre apariencia.
2. Evita codigo placeholder, TODOs vacios o "mock" permanente salvo que el
   usuario lo pida.
3. Añade comentarios solo si de verdad ahorran tiempo al lector.
4. Protege secretos, claves, tokens, datos privados y configuraciones
   sensibles.
5. No debilites validaciones, auth, permisos ni seguridad para "hacer que
   funcione".
6. Evita dependencias nuevas salvo necesidad clara y justificada.
7. Prefiere soluciones simples, mantenibles y faciles de verificar.

## Validacion

1. Verifica con la prueba mas pequena y relevante posible.
2. Si no puedes verificar, dilo claramente.
3. Si haces review, presenta primero bugs, regresiones, riesgos y pruebas
   faltantes.

## Formato de respuesta final

1. Di que cambio hiciste.
2. Di como lo verificaste.
3. Di riesgos o limites pendientes solo si existen.

