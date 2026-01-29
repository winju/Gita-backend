from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import boto3
import uuid
import os
from botocore.client import Config
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # AWS Amplify server - React code deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# AWS clients
polly = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
    config=Config(s3={'addressing_style': 'virtual'}),
    endpoint_url=f"https://s3.{os.getenv('AWS_REGION')}.amazonaws.com"
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


# Debug prints
print("=== DEBUG INFO ===")
print("AWS_REGION from env:", os.getenv("AWS_REGION"))
print("Bucket name from env:", BUCKET_NAME)
print("S3 client region:", s3.meta.region_name)
print("Polly client region:", polly.meta.region_name)
print("===================")



class Query(BaseModel):
    query: str


def generate_presigned_url(bucket_name, file_name, expiry=3600):
    """Generate a temporary signed URL for private S3 objects"""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": file_name},
        ExpiresIn=expiry
    )


@app.post("/ask")
async def ask(query: Query):
    try:

        # Step 0: Classify the query before answering
        classification = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify if the following user query is devotional/spiritual. Reply only 'devotional' or 'not devotional'."},
                {"role": "user", "content": query.query}
            ]
        )

        category = classification.choices[0].message.content.strip().lower()

        if "not devotional" in category:
            raise HTTPException(status_code=400, detail="Only devotional questions are allowed 🙏")


        # 1. Build muni-style prompt
        prompt = (
            f"You are a wise muni. Answer calmly and devotionally in Hindi. "
            f"Start and end the answer with ॐ chantation.\nUser query: {query.query}\n"
        )

        # 2. Call GPT‑4o mini
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        text_answer = completion.choices[0].message.content

        # 3. Convert text to speech using AWS Polly
        polly_response = polly.synthesize_speech(
            Text=text_answer,
            OutputFormat="mp3",
            VoiceId="Kajal",       # Female Hindi voice
            Engine="neural",       # Use Neural TTS engine
            LanguageCode="hi-IN"   # Hindi language
        )

        audio_stream = polly_response["AudioStream"].read()

        # 4. Save audio to S3
        file_name = f"{uuid.uuid4()}.mp3"

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=file_name,
            Body=audio_stream,
            ContentType="audio/mpeg"
        )

        # 5. Generate pre-signed URL (valid for 1 hour)
        audio_url = generate_presigned_url(BUCKET_NAME, file_name)
        print("Final presigned URL:", audio_url)


        # 6. Return both text + audio URL
        return {
            "text": text_answer,
            "audio_url": audio_url
        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))

