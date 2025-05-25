# smartsheet_writer.py

import smartsheet
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def overwrite_smartsheet_with_df(df: pd.DataFrame, sheet_id: str, api_token: str) -> str:
    try:
        smartsheet_client = smartsheet.Smartsheet(api_token)
        smartsheet_client.errors_as_exceptions(True)
        
        logger.info(f"Loading target sheet {sheet_id}...")

        # Load sheet
        sheet = smartsheet_client.Sheets.get_sheet(sheet_id)
        column_map = {col.title: col.id for col in sheet.columns}
        
        logger.info(f"Target sheet columns: {list(column_map.keys())}")
        logger.info(f"DataFrame columns: {list(df.columns)}")

        # Check if DataFrame columns match sheet columns
        missing_cols = [col for col in df.columns if col not in column_map]
        if missing_cols:
            logger.warning(f"DataFrame columns not found in sheet: {missing_cols}")

        # Delete all existing rows
        row_ids = [row.id for row in sheet.rows]
        logger.info(f"Deleting {len(row_ids)} existing rows...")
        
        for i in range(0, len(row_ids), 450):
            batch = row_ids[i:i+450]
            logger.info(f"Deleting batch {i//450 + 1}: {len(batch)} rows")
            smartsheet_client.Sheets.delete_rows(sheet_id, batch)

        # Prepare new rows
        rows_to_add = []
        for idx, row in df.iterrows():
            cells = []
            for col in df.columns:
                if col in column_map:
                    value = row[col]
                    # Handle NaN values
                    if pd.isna(value):
                        value = None
                    cells.append({
                        'column_id': column_map[col],
                        'value': value
                    })
            new_row = smartsheet.models.Row()
            new_row.to_bottom = True
            new_row.cells = cells
            rows_to_add.append(new_row)

        logger.info(f"Adding {len(rows_to_add)} new rows...")

        # Add rows in batches
        for i in range(0, len(rows_to_add), 450):
            batch = rows_to_add[i:i+450]
            logger.info(f"Adding batch {i//450 + 1}: {len(batch)} rows")
            result = smartsheet_client.Sheets.add_rows(sheet_id, batch)
            logger.info(f"Batch {i//450 + 1} result: {result.result_code}")

        return f"✅ Overwrote sheet {sheet_id} with {len(rows_to_add)} rows."

    except Exception as e:
        logger.error(f"Exception in overwrite_smartsheet_with_df: {str(e)}")
        return f"❌ Failed to overwrite sheet: {e}"