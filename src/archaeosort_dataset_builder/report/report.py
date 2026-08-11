import json

from src.archaeosort_dataset_builder.config.settings import settings


def report():

    reports = settings.reports

    html = []

    html.append("<html>")
    html.append("<head>")
    html.append("<title>ArchaeoSort Report</title>")
    html.append("<style>")
    html.append("body{font-family:Arial;margin:40px;background:#fafafa}")
    html.append("table{border-collapse:collapse;width:100%}")
    html.append("td,th{border:1px solid #ccc;padding:8px}")
    html.append("th{background:#eee}")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")

    html.append("<h1>ArchaeoSort Dataset Report</h1>")

    for file in reports.glob("*.json"):
        data = json.loads(file.read_text())

        html.append(f"<h2>{file.stem}</h2>")
        html.append("<pre>")
        html.append(json.dumps(data, indent=4))
        html.append("</pre>")

    html.append("</body></html>")

    output = reports / "report.html"

    output.write_text("\n".join(html), encoding="utf8")

    print(output)
