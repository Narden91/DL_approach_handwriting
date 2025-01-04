import boto3
import pandas as pd
from io import StringIO
from typing import Dict, Optional
from rich import print as rprint
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
import time
import numpy as np


class S3IOHandler:
    def __init__(self, config_handler, verbose: bool = True, max_retries: int = 5):
        """Initialize S3 IO handler with config."""
        self.config = config_handler
        self.verbose = verbose
        self.bucket_name = self.config.s3_bucket
        self.max_retries = max_retries
        self.s3_client = self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client using config credentials with improved settings."""
        credentials = self.config.get_aws_credentials()
        
        # Configure boto3 with increased timeouts and retries
        config = Config(
            connect_timeout=30,  # 30 seconds for connection
            read_timeout=60,     # 60 seconds for read
            retries=dict(
                max_attempts=5,  # Maximum number of retries
                mode='adaptive'  # Adaptive retry mode
            ),
            # Additional configurations for stability
            max_pool_connections=50,
            tcp_keepalive=True
        )
        
        return boto3.client(
            's3',
            endpoint_url=credentials['aws_endpoint_url'],
            aws_access_key_id=credentials['aws_access_key_id'],
            aws_secret_access_key=credentials['aws_secret_access_key'],
            config=config
        )

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time with jitter."""
        base_delay = min(300, 2 ** attempt)  # Cap at 300 seconds
        jitter = np.random.uniform(0, 0.1 * base_delay)  # Add 10% jitter
        return base_delay + jitter

    def load_data(self, file_key: str) -> pd.DataFrame:
        """Load DataFrame from S3 with retries."""
        if self.verbose:
            rprint(f"[blue]Loading data from S3: {file_key}[/blue]")
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
                
                # Read in chunks to handle large files
                chunks = []
                body = obj['Body']
                
                for chunk in pd.read_csv(body, chunksize=10000):
                    chunks.append(chunk)
                
                return pd.concat(chunks, ignore_index=True)
                
            except ReadTimeoutError as e:
                last_exception = e
                wait_time = self._exponential_backoff(attempt)
                
                if self.verbose:
                    rprint(f"[yellow]Read timeout on attempt {attempt + 1}/{self.max_retries}. "
                          f"Retrying in {wait_time:.1f} seconds...[/yellow]")
                
                time.sleep(wait_time)
                
                # Reinitialize client on timeout
                self.s3_client = self._init_s3_client()
                
            except ClientError as e:
                last_exception = e
                if e.response['Error']['Code'] in ['NoSuchKey', 'NoSuchBucket']:
                    rprint(f"[red]S3 error: {e.response['Error']['Message']}[/red]")
                    raise
                
                wait_time = self._exponential_backoff(attempt)
                if self.verbose:
                    rprint(f"[yellow]S3 client error on attempt {attempt + 1}/{self.max_retries}. "
                          f"Retrying in {wait_time:.1f} seconds...[/yellow]")
                
                time.sleep(wait_time)
            
            except Exception as e:
                last_exception = e
                rprint(f"[red]Unexpected error: {str(e)}[/red]")
                raise

        # If we've exhausted all retries, raise the last exception
        raise last_exception or Exception("Failed to load data after all retries")

    def save_data(self, df: pd.DataFrame, file_key: str) -> None:
        """Save DataFrame to S3 with retries."""
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    rprint(f"[blue]Saving data to S3: {file_key}[/blue]")
                
                # Convert to CSV buffer
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=file_key,
                    Body=csv_buffer.getvalue()
                )
                
                if self.verbose:
                    rprint(f"[bold green]Data saved to s3://{self.bucket_name}/{file_key}[/bold green]")
                return
                
            except (ClientError, ReadTimeoutError) as e:
                if attempt == self.max_retries - 1:
                    rprint(f"[red]Failed to save data to S3 after {self.max_retries} attempts: {e}[/red]")
                    raise
                
                wait_time = self._exponential_backoff(attempt)
                if self.verbose:
                    rprint(f"[yellow]Error on attempt {attempt + 1}/{self.max_retries}. "
                          f"Retrying in {wait_time:.1f} seconds...[/yellow]")
                
                time.sleep(wait_time)
                # Reinitialize client on error
                self.s3_client = self._init_s3_client()

# class S3IOHandler:
#     def __init__(self, config_handler: ConfigHandler, verbose: bool = True):
#         """Initialize S3 IO handler with config."""
#         self.config = config_handler
#         self.verbose = verbose
#         self.bucket_name = self.config.s3_bucket
#         self.s3_client = self._init_s3_client()

#     def _init_s3_client(self):
#         """Initialize S3 client using config credentials."""
#         credentials = self.config.get_aws_credentials()
#         return boto3.client(
#             's3',
#             endpoint_url=credentials['aws_endpoint_url'],
#             aws_access_key_id=credentials['aws_access_key_id'],
#             aws_secret_access_key=credentials['aws_secret_access_key']
#         )

#     def load_data(self, file_key: str) -> pd.DataFrame:
#         """Load DataFrame from S3."""
#         if self.verbose:
#             rprint(f"[blue]Loading data from S3: {file_key}[/blue]")
        
#         obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
#         return pd.read_csv(obj['Body'])

#     def save_data(self, df: pd.DataFrame, file_key: str) -> None:
#         """Save DataFrame to S3."""
#         try:
#             if self.verbose:
#                 rprint(f"[blue]Saving data to S3: {file_key}[/blue]")
            
#             csv_buffer = StringIO()
#             df.to_csv(csv_buffer, index=False)
            
#             self.s3_client.put_object(
#                 Bucket=self.bucket_name,
#                 Key=file_key,
#                 Body=csv_buffer.getvalue()
#             )
            
#             if self.verbose:
#                 rprint(f"[bold green]Data saved to s3://{self.bucket_name}/{file_key}[/bold green]")
#         except Exception as e:
#             rprint(f"[red]Failed to save data to S3: {e}[/red]")