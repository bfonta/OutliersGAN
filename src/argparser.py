def add_args(parser):
    parser.add_argument(
        '--checkpoint',
        type=int,
        default=0,
        help='Checkpoint prefix number.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Mode to run the model.'
    )
    parser.add_argument(
        '--fname',
        type=str,
        default='train',
        help='Input file name.'
    )
    parser.add_argument(
        '--noGPU',
        type=int,
        default=0,
        help='Prevents Tensorflow from using GPU.'
    )
    parser.add_argument(
        '--data_type',
        type=str,
        default=0,
        help='gal_form_burst or qso_zWarning'
    )
    return parser.parse_known_args()
