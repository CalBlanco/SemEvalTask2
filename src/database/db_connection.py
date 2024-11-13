import sqlite3 as sql
import os

db_path = os.path.join(os.path.dirname(__file__), '../../database/entities.db')


def command(query_string:str):
    '''Simple function to wrap database functionality [Executes raw SQL be careful]'''
    con = sql.connect(db_path)
    cur = con.cursor()

    res = cur.execute(query_string)
    res = res.fetchall()

    con.close()

    return res


def query(projections:str, conditions:str=''):
    '''Slightly more handled wrapper that manages the table and organization of the query fairly free form'''
    res = command(f'''
        SELECT {projections} FROM entity_translation {conditions}
    ''')
    return res

def query_by_name(name:str, projections='*', additional=''):
    '''Wrapped function for querying by name, optionally pass projections'''
    res = command(f'''
        SELECT {projections} FROM entity_translation WHERE name like "{name}" {additional}
    ''')

    return res

def query_by_instance(instance:str, projections='*', additional=''):
    '''Wrapped function for querying by instance, optionally pass projections'''

    res = command(f'''
        SELECT {projections} FROM entity_translation WHERE instance_of like "{instance}" {additional}
    ''')
    return res

