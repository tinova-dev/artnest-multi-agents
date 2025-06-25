from pydantic import BaseModel

class ImageURLs(BaseModel):
    image_url: str
    website_link: str