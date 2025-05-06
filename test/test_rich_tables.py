import json

from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import DoclingDocument, RefItem, TableData


def test_construct_rich_table():
    num_cols = 3
    num_rows = 3

    # Construct a table by building data without cells first...
    doc = DoclingDocument(name="test_rich_table")
    data = TableData(num_rows=num_rows, num_cols=num_cols)
    tbl = doc.add_table(data=data)

    # Then add cells with the update_cell method.
    first_cell = tbl.update_cell(
        text="AB",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=2,
        column_header=False,
        row_header=True,
    )

    print(first_cell)

    second_cell = tbl.update_cell(
        text="C",
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=2,
        end_col_offset_idx=3,
        column_header=False,
        row_header=True,
    )

    print(second_cell)

    third_cell = tbl.update_cell(
        text="1",
        start_row_offset_idx=1,
        end_row_offset_idx=3,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
        column_header=False,
        row_header=True,
    )

    print(third_cell)

    fourth_cell = tbl.update_cell(
        text="2",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=1,
        end_col_offset_idx=2,
        column_header=False,
        row_header=False,
    )

    print(fourth_cell)

    fifth_cell = tbl.update_cell(
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=2,
        end_col_offset_idx=3,
        column_header=False,
        row_header=False,
    )

    print(fifth_cell)

    # Add a child item to the fifth cell to add rich content.
    doc.add_text(parent=fifth_cell, text="Foo", label=DocItemLabel.TEXT)

    # Test serialization
    print(tbl.export_to_markdown(doc))
    print(tbl.export_to_html(doc))
    print(json.dumps(doc.export_to_dict(), indent=2))

    # Test resolution from table cell ref:
    resolved = RefItem(cref="#/tables/0/data/table_cells/2").resolve(doc)
    print(resolved)
