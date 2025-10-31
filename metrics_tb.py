import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_tensorboard_data_wide(log_dir, output_csv):
    """
    Extracts scalar data from a TensorBoard log directory and saves it to a
    'wide' format CSV, where each metric is a column.
    """
    
    try:
        accumulator = EventAccumulator(log_dir)
        accumulator.Reload()
    except Exception as e:
        print(f"Error loading log directory '{log_dir}': {e}")
        return

    available_tags = accumulator.Tags().get('scalars', [])
    if not available_tags:
        print(f"No scalar data found in '{log_dir}'.")
        return

    print(f"Found scalar tags: {available_tags}")

    all_data = []

    # Extract data in "long" format
    for tag in available_tags:
        events = accumulator.Scalars(tag)
        for event in events:
            all_data.append({
                'tag': tag,
                'step': event.step,
                'wall_time': event.wall_time,
                'value': event.value
            })

    if not all_data:
        print("No data points found.")
        return

    # Convert to DataFrame
    df_long = pd.DataFrame(all_data)

    # --- Pivot the data to "wide" format ---
    # We use 'step' as the index and 'tag' as the columns.
    # 'value' will be the data in the cells.
    try:
        # Use pivot_table to handle potential duplicate steps/tags (if any)
        # Using 'mean' as aggfunc, though 'last' or 'first' might also be suitable
        df_wide = df_long.pivot_table(
            index='step', 
            columns='tag', 
            values='value',
            aggfunc='mean'  # or 'last'
        )
        
        # We can also pivot wall_time if needed, but it's often complex
        # as each tag might have a slightly different timestamp.
        # For simplicity, we'll focus on 'value' as requested.

    except Exception as e:
        print(f"Error pivoting data: {e}")
        print("This can happen if one metric is logged at a step where "
              "another is not. We will try to fill NaNs.")
        # Simpler pivot, which is often sufficient
        df_wide = df_long.pivot(index='step', columns='tag', values='value')


    # Reset index to make 'step' a regular column
    df_wide = df_wide.reset_index()
    # Rename the column index (if any) to be flat
    df_wide.columns.name = None
    
    # At this point, columns will be: 'step', 'metric_1', 'metric_2', etc.

    # Save to CSV
    df_wide.to_csv(output_csv, index=False)
    print(f"\nSuccessfully extracted and pivoted data to '{output_csv}'")
    
    print("\nPivoted Data Head:")
    print(df_wide.head())

# --- Main execution ---
if __name__ == "__main__":
    log_directory = "./lightning_logs/version_0"
    output_file = "tensorboard_export_wide.csv"
    
    export_tensorboard_data_wide(log_directory, output_file)