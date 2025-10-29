from scapy.all import *
import random

def fuzz2(p, p_prev=False,  # type: _P
         _inplace=0,  # type: int
         ):
    # type: (...) -> _P
    """
    Transform a layer into a fuzzy layer by replacing some default values
    by random objects.

    :param p: the Packet instance to fuzz
    :return: the fuzzed packet.
    """
    if not p_prev:
        #print('no prev')
        #print('TOTAL BYTES', len(bytes(p)))
        pkt = type(p)(bytes(fuzz(p)))
        if UDP in pkt:
            del pkt[UDP].chksum
            pkt = UDP(raw(pkt))
        return pkt

    if not _inplace:
        p = p.copy()
    q = cast(Packet, p)

    '''We can simplify this code on the assumption theres only one layer and '''
    assert isinstance(q.payload, NoPayload), q

    new_values = {}
    for f, f_prev in zip(q.fields_desc, p_prev.fields_desc):
        #print(f, f_prev)
        assert f.name==f_prev.name, (f, f_prev)

        if f.name in q.fields:
            continue  # Skip fields that are already set

        new_values[f.name] = randval2(f, p_prev)
    
    q.default_fields.update(new_values)

    #print('TOTAL BYTES', len(bytes(q)))
    q = type(q)(bytes(q))

    return q


def randval2(field, prev_pkt: Packet):
    #print('here1',field)
    if isinstance(field, MultipleTypeField):
        field = field._find_fld_pkt(prev_pkt)
        print('multitypefield',field)


    valid_fields = ['StrField', 'Field', 'ByteField', 'ByteEnumField', 'StrLenField', 'FlagsField', 'StrFixedLenField', 'PadField', 
                    'SockAddrsField', 'LenField', 'EnumField', 'FieldLenField', 'MayEnd', 'XStrLenField', 'IP6Field', 'UUIDField', 'ShortField', 'XByteField', 'XShortField', 
                    'BitField', 'BitEnumField', 'MACField', 'FieldListField', 'XIntField', 'IntField', 'LongField', 'IntEnumField', 'SignedIntField', 
                    'XNBytesField', 'XBitField', 'ShortEnumField', 'XShortEnumField', 'SourceMACField', 'SourceIPField', 'IPField']
    #fields to keep the same
    same = ['BitField', 'FlagsField', 'XBitField']

    enums = ['ShortEnumField','XShortEnumField', 'ByteEnumField', 'IntEnumField', 'EnumField', 'BitEnumField']

    markov_pos = ['ByteField', 'ShortField', 'XByteField', 'XShortField', 
                'XIntField', 'IntField', 'LongField', 'XNBytesField']
    
    markov_int = ['SignedIntField']
    
    prev_value = prev_pkt.getfieldval(field.name)

    TRANSITION = 0.6
    RANGE = (-5, 5)

    ftype =  field.__class__.__name__
    #print(f'{ftype=}')

    if ftype in same:
        return prev_value
    
    elif ftype in enums:
        return random.choice(list(field.i2s.keys()))
    
    elif ftype in markov_pos:
        if random.random() < TRANSITION:
            return min(abs(prev_value + random.randint(*RANGE)), max_val_for_field(field))
        else:
            return prev_value

    
    elif ftype in markov_int:
        if random.random() < TRANSITION:
            return min(prev_value + random.randint(*RANGE), max_val_for_field(field))
        else:
            return prev_value

    elif ftype in valid_fields:
        if random.random() < TRANSITION:
            return field.randval()
        else:
            return prev_value
    
    else:
        print('INVALID FIELD', field)
        return 1/0
    

def recalculate_lengths(pkt: Packet) -> Packet:
    for f in pkt.fields_desc:
        if isinstance(f, FieldLenField):
            print('Del', f)
            #del q.fields[f.name]
            pkt.fields.pop(f.name, None)  # removes 'ttl' if set
    # Rebuild full packet from raw bytes to trigger recalculation
    print(pkt.fields)
    return type(pkt)(bytes(pkt))



def max_val_for_field(f):
    from struct import calcsize
    import struct

    # Map struct format to max values
    fmt_max = {
        'B': 0xFF,          # unsigned char
        'H': 0xFFFF,        # unsigned short
        'I': 0xFFFFFFFF,    # unsigned int
        'Q': 0xFFFFFFFFFFFFFFFF,  # unsigned long long
        'b': 0x7F,          # signed char
        'h': 0x7FFF,        # signed short
        'i': 0x7FFFFFFF,    # signed int
        'q': 0x7FFFFFFFFFFFFFFF,  # signed long long
    }

    if hasattr(f, 'fmt'):
        if f.fmt in fmt_max:
            return fmt_max[f.fmt]
    
    name = f.__class__.__name__ 

    if 'Byte' in name:
        return 0xFF
    elif 'Short' in name:
        return 0xFFFF
    elif 'Int' in name:
        return 0xFFFFFFFF
    elif 'Long' in name:
        return 0xFFFFFFFFFFFFFFFF
    
    print('UNKNOWNMAX')
    return None  # unknown max
