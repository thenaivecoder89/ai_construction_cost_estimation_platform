select * from information_schema.tables where table_schema = 'public';

select * from documents;

select * from takeoff_layer;

select count(*) from takeoff_layer where project_id = 'private_villa_001';

delete from takeoff_layer where project_id = 'private_villa_001';

select * from extracted_text;

select * from extracted_tables;

select * from extracted_table_rows;

select count(*) from chunks where embedding is null;

select * from build_document_inventory where source_group = 'client_data';

select * from cost_database;

-- delete data sql prompts - DO NOT TOUCH

delete from extracted_text;

delete from extracted_tables;

delete from extracted_table_rows;

delete from chunks;

delete from build_document_inventory;

delete from documents;

delete from cost_database;

select * from project_uploads;

-- select table data
