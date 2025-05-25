from fastapi import FastAPI, Request
from smartsheet_writer import overwrite_smartsheet_with_df
import smartsheet
import pandas as pd

app = FastAPI()

@app.post("/run-task")
async def run_task(request: Request):
    body = await request.json()
    
    api_token = body.get("api_token")
    source_sheet_id = body.get("source_sheet_id")
    target_sheet_id = body.get("target_sheet_id")

    if not all([api_token, source_sheet_id, target_sheet_id]):
        return {"status": "error", "message": "Missing required fields."}

    try:
        # Step 1: Read source Smartsheet
        client = smartsheet.Smartsheet(api_token)
        source_sheet = client.Sheets.get_sheet(source_sheet_id)
        data = []

        # Convert rows to DataFrame
        for row in source_sheet.rows:
            row_data = {}
            for cell in row.cells:
                col = next((col.title for col in source_sheet.columns if col.id == cell.column_id), None)
                if col:
                    row_data[col] = cell.value
            data.append(row_data)
        df = pd.DataFrame(data)

        # Step 2: Preprocess the DataFrame
        # TODO: Customize this to your preprocessing logic
        df = df.dropna()  # Simple example

        # Step 3: Overwrite target sheet
        result = overwrite_smartsheet_with_df(df, int(target_sheet_id), api_token)
        return {"status": "success", "message": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
