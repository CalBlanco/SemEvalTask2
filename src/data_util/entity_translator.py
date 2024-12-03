from database import query_by_name

def translate_entity(entity_name:str, desired_translations:list[str]):
    """Get translations for an entity
    
    **ARGS**
        - entity_name: The name of the entitiy we want to lookup for a translation
        - desired_translations: The list of desired translations i.e ['ar', 'de'] would get translations for arabic and german

    **RETURNS**
        a tuple containing the translations in the order specified by desired_translations 
    """
    q = query_by_name(entity_name, ','.join(desired_translations))
    if len(q) > 0:
        return q[0]
    else:
        return []
   

def translate_entities(results:list[list[tuple[str,str]]], desired_translations:list[str]=['ar', 'de', 'es', 'fr', 'it', 'ja']):
    """ Translate the outputs from the NER accuracies into desired translation languages

    **ARGS**
        - results: a List of a list of tuples where each tuple represents the name of the entity and the type
        - desired_translations: a list of desired translations 
    
    **RETURNS**
        Returns a list of tuples containing (entity_name, ...translations)
    """
    ret = []
    for pred in results:
        for name,_ in pred:
            translation_info = translate_entity(name, desired_translations) 
            ret.append((name, *translation_info))
    
    return ret