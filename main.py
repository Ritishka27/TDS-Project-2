from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import asyncio
from playwright.async_api import async_playwright
import pdfplumber
import requests
import tempfile

# -------------------------------------
# SETTINGS
# -------------------------------------
EXPECTED_SECRET = "your-secret-here"

app = FastAPI()


# -------------------------------------
# incoming payload format
# -------------------------------------
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


# -------------------------------------
# MAIN ENDPOINT
# -------------------------------------
@app.post("/quiz")
async def quiz_handler(payload: QuizRequest):
    # 1. Validate secret
    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 2. Visit quiz page & extract instructions
    question, pdf_url, submit_url = await scrape_quiz_page(payload.url)

    # 3. Solve (example: PDF sum problem)
    answer = solve_pdf_sum_question(pdf_url)

    # 4. Submit answer to the given submit URL
    submit_payload = {
        "email": payload.email,
        "secret": payload.secret,
        "url": payload.url,
        "answer": answer
    }

    res = requests.post(submit_url, json=submit_payload)
    return res.json()


# -------------------------------------
# SCRAPE QUIZ PAGE USING PLAYWRIGHT
# -------------------------------------
async def scrape_quiz_page(quiz_url: str):
    """
    Extract:
    - the question text
    - the PDF download link
    - the submit URL
    """

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(quiz_url)

        # Example: the question text is in #result after JS runs
        question = await page.locator("#result").inner_text()

        # Extract PDF download link — sample selector
        pdf_url = await page.locator("a[href$='.pdf']").get_attribute("href")

        # Submit URL embedded inside the page
        submit_url = await page.locator("pre").inner_text()
        # Find URL in JSON-like block
        import json, re
        submit_json = json.loads(re.findall(r"\{.*\}", submit_url, re.S)[0])
        submit_url = submit_json.get("submit_url", "https://example.com/submit")  # replace logic

        await browser.close()
        return question, pdf_url, submit_url


# -------------------------------------
# SOLVER: Example for “sum value column on page 2”
# -------------------------------------
def solve_pdf_sum_question(pdf_url: str):
    # Download the PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(requests.get(pdf_url).content)
    temp_file.close()

    # Read page 2 and sum "value" column
    with pdfplumber.open(temp_file.name) as pdf:
        page2 = pdf.pages[1]  # page 2 = index 1
        table = page2.extract_table()

    # Identify "value" column
    header = table[0]
    value_idx = header.index("value")

    # Sum the column
    total = 0
    for row in table[1:]:
        total += float(row[value_idx])

    return total
