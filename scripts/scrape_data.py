import os
import csv
import asyncio
from telethon.sync import TelegramClient
from telethon.errors import ChannelPrivateError, FloodWaitError, UsernameInvalidError
from dotenv import load_dotenv
import pandas as pd

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir, existing_ids):
    try:
        print(f"\n📡 Fetching entity for channel: {channel_username}")
        entity = await client.get_entity(channel_username)
        channel_title = entity.title

        print(f"📥 Scraping messages from: {channel_title} ({channel_username})")

        async for message in client.iter_messages(entity, limit=10000):
            message_key = (channel_username, message.id)
            if message_key in existing_ids:
                print(f"🔁 Skipping already scraped message: {message.id} from {channel_username}")
                continue

            media_path = ""
            text = ""

            try:
                # Extract text content
                if message.message:
                    text += message.message.strip()
                if message.media and hasattr(message.media, 'caption') and message.media.caption:
                    text += f"\n{message.media.caption.strip()}"

                # Download image if present
                if message.media and hasattr(message.media, 'photo'):
                    filename = f"{channel_username}_{message.id}.jpg"
                    media_path = os.path.join('data', 'media', filename)
                    await client.download_media(message.media, os.path.join(media_dir, filename))
                    print(f"🖼️ Downloaded media: {filename}")

                # Write message data
                writer.writerow([
                    channel_title,
                    channel_username,
                    message.id,
                    text,
                    str(message.date),
                    media_path
                ])
            except Exception as msg_err:
                print(f"⚠️ Error processing message {message.id} in {channel_username}: {msg_err}")
                continue

        print(f"✅ Finished scraping: {channel_title} ({channel_username})")

    except ChannelPrivateError:
        print(f"🚫 Channel '{channel_username}' is private or access denied.")
    except UsernameInvalidError:
        print(f"❌ Channel username '{channel_username}' is invalid.")
    except FloodWaitError as e:
        print(f"⏱️ Rate limit hit. Sleeping for {e.seconds} seconds.")
        await asyncio.sleep(e.seconds)
        await scrape_channel(client, channel_username, writer, media_dir, existing_ids)
    except Exception as e:
        print(f"❌ Unexpected error with channel {channel_username}: {e}")


# Get credentials from .env
def get_client():
    load_dotenv()
    api_id = int(os.getenv('API_ID'))
    api_hash = os.getenv('API_HASH')
    return TelegramClient('scraping_session', api_id, api_hash)


# Main async runner
async def main():
    print("🚀 Starting Telegram scraping process...")
    client = get_client()
    await client.start()
    print("✅ Telegram client started.")

    # Directory setup
    base_dir = os.getenv('BASE_DIR')
    data_dir = os.path.join(base_dir, 'data/raw/')
    media_dir = os.path.join(base_dir, 'data/media/')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    # CSV output file
    csv_path = os.path.join(data_dir, 'telegram_data.csv')
    print(f"📄 Writing scraped data to: {csv_path}")

    # Check for already scraped message IDs
    existing_ids = set()
    write_header = True
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            existing_ids = set(zip(df_existing['Channel Username'], df_existing['Message ID'].astype(int)))
            write_header = False  # Don't rewrite the header
            print(f"🔎 Found {len(existing_ids)} previously scraped messages.")
        except Exception as e:
            print(f"⚠️ Could not load existing CSV: {e}")

    # Open file in append mode
    with open(csv_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Channel Title', 'Channel Username', 'Message ID', 'Message Text', 'Date', 'Media Path'])

        # Channel list (you can add more as needed)
        channels = [
            # '@Shageronlinestore',
            '@ZemenExpress',
            '@nevacomputer',
            '@meneshayeofficial',
            '@ethio_brand_collection',
            '@Leyueqa',
            '@sinayelj',
            '@Shewabrand',
            '@helloomarketethiopia',
            '@modernshoppingcenter',
            '@qnashcom',
            '@Fashiontera',
            '@kuruwear',
            '@gebeyaadama',
            '@MerttEka',
            '@forfreemarket',
            '@classybrands',
            '@marakibrand',
            '@aradabrand2',
            '@marakisat2',
            '@belaclassic',
            '@AwasMart',
            'qnashcom'
        ]

        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir, existing_ids)

    print("🎉 All channels processed!")


# For Notebook use
async def run_main():
    await main()

if __name__ == '__main__':
    asyncio.run(main())