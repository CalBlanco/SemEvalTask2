import sqlite3 as sql
import os

db_path = os.path.join(os.path.dirname(__file__), '../../database/ent.db')


def command(query_string:str)->list[tuple]:
    """Raw SQL Executor
    
    ARGS
        **query_string** -- A string of raw sql to execute on the database 

    RETURNS
        A list of results
    NOTES
        Will literally let you do anything be careful
        It opens and closes the SQL connection so you do not have to 
    """
    con = sql.connect(db_path)
    cur = con.cursor()

    res = cur.execute(query_string)
    res = res.fetchall()

    con.close()

    return res


def query(projections:str, conditions:str='')->list[tuple]:
    """Simple Query function over the entity_translation table
    
    ARGS
        **projections** -- A string containing column projections i.e 'name, instance_of' or 'name, ar' [Must be a string for now]
        **conditions**  -- A string for selection conditions i.e WHERE name == "Bobby Hill" (default none)
    
    RETURNS
        A list of query results
    NOTES
        Can technically pass more than conditions bc this is just raw string (SQL Injectionable)
    """
    res = command(f'''
        SELECT {projections} FROM entity_translation {conditions}
    ''')
    return res

def query_by_name(name:str, projections='*', additional='')->list[tuple]:
    """Query over the table specifically by name
    
    ARGS
        **name**        -- Literal name or pattern to match against for names 
        **projections** -- Columns you would like projected (default all columns are returned)
        **additional**  -- Any additional SQL you want to add on to the end of the function call (default none)

    RETURNS
        A list of query results
    EXAMPLES
        1. Query a particular name: `query_by_name("John Smith")` -> [(wiki_id, name, instance_of, ... translations)]
        2. Query over a pattern: `query_by_name("Jo%")` -> [(names that start with 'Jo')]
        3. Query over a pattern projecting only the name and arabic translation: `query_by_name`("Jo%", "name, ar")
    """
    res = command(f'''
        SELECT {projections} FROM entity_translation WHERE name like "{name}" {additional}
    ''')

    return res

def query_by_instance(instance:str, projections='*', additional='')->list[tuple]:
    """Essentially `query_by_name` but for matching patterns in the instance_of column
    
    ARGS
        **instance**    -- A string or pattern to match for the instance_of column
        **projections** -- A string of columns to project (default all)
        **additional**  -- Raw sql to additionally pass like conditionals or aggregations (default none) 

    RETURNS
        A list of query results
    NOTES
        See the query_by_name for use case examples
    """

    res = command(f'''
        SELECT {projections} FROM entity_translation WHERE instance_of like "{instance}" {additional}
    ''')
    return res


def query_by_id(ids:list, projections='*', additional='')->list[tuple]:
    """Perform a query by passing in a list of wiki_ids
    
    ARGS
        **ids** -- A list of wiki_ids to query for i.e ['Q49', 'Q1234']
        **projections** -- A string of columns to project (default all)
        **additional** -- Additional sql to pass into the request (default none)

    RETURNS
        A list of tuples representing the query results 
    """
    ids = [f'\"{x}\"' for x in ids] #annoying requirement to wrap the ids in quotes
    q_string = f'''SELECT {projections} FROM entity_translation WHERE wiki_id IN ({", ".join(ids)}) {additional}'''
    res = command(q_string)

    return res