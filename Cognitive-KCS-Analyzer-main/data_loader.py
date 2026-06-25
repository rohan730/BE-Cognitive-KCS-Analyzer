import pandas as pd

def load_csv_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV data from {file_path}: {e}")
        return None

if __name__ == '__main__':
    articles_df = load_csv_data('sample_data/kcs_articles.csv')
    if articles_df is not None:
        print("\nSample KCS Articles:")
        print(articles_df.head())

    tickets_df = load_csv_data('sample_data/support_tickets.csv')
    if tickets_df is not None:
        print("\nSample Support Tickets:")
        print(tickets_df.head())
