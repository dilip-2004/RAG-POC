# import pymupdf4llm

# md_test = pymupdf4llm.to_markdown("my_docs/717821p213 - DILIP S.pdf")
# print(f"docs = {md_test}")
#
# import pathlib
# pathlib.Path("output.md").write_bytes(md_test.encode())

# import pymupdf4llm
# llama_reader = pymupdf4llm.LlamaMarkdownReader()
# llama_docs = llama_reader.load_data("my_docs/717821p213 - DILIP S.pdf")
# print(f"llm_docs = {llama_docs}")

import pymupdf
doc = pymupdf.open("my_docs/717821p213 - DILIP S.pdf")

out = open("Documents/output1.txt", "wb")
for page in doc:
    text = page.get_text().encode()
    print(text)
    out.write(text)

out.close()
print("completed")