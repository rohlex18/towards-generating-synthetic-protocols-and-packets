from scapy.all import *

def get_packet_instance_by_name(name):
    if name == 'ARP':
        return ARP(hwtype = 0x1, ptype  = 0x0800, hwlen=6, plen=4)
    elif name == 'VRRP':
        return VRRP(addrlist=[], ipcount=0)
    else:
        return get_packet_class_by_name(name)()

def get_packet_class_by_name(name):
    to_check = [Packet]
    seen = set()
    while to_check:
        cls = to_check.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if cls.__name__ == name:
            return cls
        to_check.extend(cls.__subclasses__())
    return None

def first_n_fields_bytes(pkt, n):
    result = b""
    fields = pkt.fields_desc[:n]
    for field in fields:
        # Get the actual value in this packet instance
        val = getattr(pkt, field.name)
        # Pack it into bytes (field.i2m handles internal-to-machine representation)
        # then field.addfield adds it to a bytes buffer
        result = field.addfield(pkt, result, val)
    return result

def pkt2field_names_bytes(layers, par, p):
    c = get_packet_class_by_name(layers[0]) #class of layer
    if c is None:
        raise ValueError(f"get_packet_class_by_name returned None for {layers[0]}")
    num_fields = len(c.fields_desc)
    for layer in layers:
        if p.haslayer(layer):
            p = p.getlayer(layer)
            p.remove_payload()
            return [field.name for field in p.fields_desc[:num_fields]], first_n_fields_bytes(p, num_fields)
        
    #try parse manually
    try:
        raw = bytes(p[par].payload)
        p = c(raw)
        p.remove_payload()
        return [field.name for field in p.fields_desc], bytes(p)

    except Exception as e:
        print('SCAPY TOOLS FAILED', e, layers, p)
        return False, False

def get_field_offsets(layer: Packet):
    offsets = []
    current_offset = 0
    bitfield_group = []
    
    def flush_bitfield_group():
        nonlocal current_offset
        if not bitfield_group:
            return
        total_bits = sum(f.size for f in bitfield_group)
        byte_len = (total_bits + 7) // 8
        for _ in bitfield_group:
            offsets.append(current_offset)
        current_offset += byte_len
        bitfield_group.clear()
    
    for field in layer.fields_desc:
        if isinstance(field, BitField):
            bitfield_group.append(field)
        else:
            flush_bitfield_group()
            try:
                val = layer.getfieldval(field.name)
                raw = field.addfield(layer, b"", val)
                offsets.append(current_offset)
                current_offset += len(raw)
            except Exception as e:
                print(f"Skipping field '{field.name}': {e}")
                offsets.append(None)

    flush_bitfield_group()
    return offsets

def find_valid_headers():
    low_layers = [Ether, IP, TCP, UDP]
    for ll in low_layers:
        for l in set([l[1] for l in ll.payload_guess]):
            try:
                fields = [type(c).__name__ for c in l.fields_desc]
                if all(elem in valid_fields for elem in fields) and len(fields)>2:
                    print(l.__name__, ll)
                    print(l.fields_desc)
            except Exception as e:
                print(e, l)

def pkt2field_types(layers):
    p = get_packet_instance_by_name(layers[0])
    res = []
    for field in p.fields_desc:
        if isinstance(field, MultipleTypeField):
            field = field._find_fld_pkt(p)
            print('multitypefield',field)
        res.append(field.__class__.__name__)
    return res