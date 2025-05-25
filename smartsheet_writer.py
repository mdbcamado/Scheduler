# smartsheet_writer.py

import smartsheet
import pandas as pd

def overwrite_smartsheet_with_df(df: pd.DataFrame, sheet_id: str, api_token: str) -> str:
    try:
        smartsheet_client = smartsheet.Smartsheet(api_token)
        smartsheet_client.errors_as_exceptions(True)

        # Load sheet
        sheet = smartsheet_client.Sheets.get_sheet(sheet_id)
        column_map = {col.title: col.id for col in sheet.columns}

        # Delete all existing rows
        row_ids = [row.id for row in sheet.rows]
        for i in range(0, len(row_ids), 450):
            smartsheet_client.Sheets.delete_rows(sheet_id, row_ids[i:i+450])

        # Prepare new rows
        rows_to_add = []
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                if col in column_map:
                    cells.append({
                        'column_id': column_map[col],
                        'value': row[col]
                    })
            new_row = smartsheet.models.Row()
            new_row.to_bottom = True
            new_row.cells = cells
            rows_to_add.append(new_row)

        # Add rows in batches
        for i in range(0, len(rows_to_add), 450):
            smartsheet_client.Sheets.add_rows(sheet_id, rows_to_add[i:i+450])

        return f"✅ Overwrote sheet {sheet_id} with {len(rows_to_add)} rows."

    except Exception as e:
        return f"❌ Failed to overwrite sheet: {e}"
